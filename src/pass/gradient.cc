/*!
 *  Copyright (c) 2016 by Contributors
 * \file gradients.cc
 * \brief Passes that takes gradient of the graph
 * This code code was modified based on mxnet codebase by Min Lin
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <functional>
#include <unordered_map>

namespace nnvm {
namespace pass {
namespace {

// default aggregate gradient function
// require operator __zero__ and __ewise_sum__ to be presented.
NodeEntry DefaultAggregateGradient(const std::vector<NodeEntry>& v) {
  if (v.size() == 1) {
    return v[0];
  } else if (v.size() == 0) {
    NodePtr zero_node = Node::Create();
    zero_node->attrs.op = Op::Get("__zero__");
    return NodeEntry{zero_node, 0, 0};
  } else {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = Op::Get("__ewise_sum__");
    sum_node->inputs = v;
    return NodeEntry{sum_node, 0, 0};
  }
}

// helper entry
struct GradEntry {
#ifdef _MSC_VER
  NodeEntry sum = NodeEntry{nullptr, 0, 0};
#else
  NodeEntry sum{nullptr, 0, 0};
#endif
  std::vector<NodeEntry> grads;
};

Graph Gradient(Graph src) {
  using nnvm::FGradient;
  using MirrorFun = std::function<int (const Node& node)>;

  CHECK_NE(src.attrs.count("grad_ys"), 0)
      << "Gradient require grad_ys to be presented.";
  CHECK_NE(src.attrs.count("grad_ys_out_grad"), 0)
      << "Gradient require grad_ys_out_grad to be presented.";
  CHECK_NE(src.attrs.count("grad_xs"), 0)
      << "Gradient require grad_xs to be presented.";
  const std::vector<NodeEntry>& ys =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys");
  const std::vector<NodeEntry>& ys_out_grad =
      src.GetAttr<std::vector<NodeEntry> >("grad_ys_out_grad");
  const std::vector<NodeEntry>& xs =
      src.GetAttr<std::vector<NodeEntry> >("grad_xs");
  using AggFun = std::function<NodeEntry (const std::vector<NodeEntry>& inputs)>;
  AggFun agg_fun = DefaultAggregateGradient;
  if (src.attrs.count("grad_aggregate_fun") != 0) {
    agg_fun = src.GetAttr<AggFun>("grad_aggregate_fun");
  }
  MirrorFun mirror_fun = nullptr;
  if (src.attrs.count("grad_mirror_fun") != 0) {
    mirror_fun = src.GetAttr<MirrorFun>("grad_mirror_fun");
  }

  // topo sort
  std::vector<NodePtr> topo_order;
  std::unordered_map<Node*, std::vector<GradEntry> > output_grads;
  DFSVisit(ys, [&](const NodePtr& node) {
      if (output_grads.count(node.get()) == 0) {
        output_grads[node.get()].resize(node->num_outputs());
      }
      topo_order.push_back(node);
    });

  CHECK_EQ(ys.size(), ys_out_grad.size());
  for (size_t i = 0; i < ys.size(); ++i) {
    output_grads[ys[i].node.get()][ys[i].index].grads = { ys_out_grad[i] };
  }

  // construct mirror reduece memory strategy if needed
  std::unordered_map<Node*, NodePtr> mirror_map;
  if (mirror_fun != nullptr) {
    for (const NodePtr& n : topo_order) {
      if (mirror_fun(*n)) {
        NodePtr new_node = Node::Create();
        *new_node = *n;
        new_node->attrs.name += "_mirror";
        for (auto& e : new_node->inputs) {
          e.node = mirror_map.at(e.node.get());
        }
        for (auto& n : new_node->control_deps) {
          n = mirror_map.at(n.get());
        }
        mirror_map[n.get()] = std::move(new_node);
      } else {
        mirror_map[n.get()] = n;
      }
    }
  }

  // traverse backward
  static auto& grad_fun_map = Op::GetAttr<FGradient>("FGradient");
  std::vector<NodeEntry> out_agg_grads;
  for (auto rit = topo_order.rbegin(); rit != topo_order.rend(); ++rit) {
    const NodePtr ptr = *rit;
    if (ptr->is_variable()) continue;
    out_agg_grads.clear();
    for (GradEntry& e : output_grads.at(ptr.get())) {
      e.sum = agg_fun(e.grads);
      out_agg_grads.push_back(e.sum);
    }
    const std::vector<NodeEntry>& input_grads = grad_fun_map[ptr->op()]
        (mirror_map.size() == 0 ? ptr : mirror_map.at(ptr.get()), out_agg_grads);
    CHECK_EQ(input_grads.size(), ptr->inputs.size())
        << "Gradient function not returning enough gradient";
    for (size_t i = 0; i < ptr->inputs.size(); ++i) {
      const NodePtr in_node = ptr->inputs[i].node;
      const uint32_t index = ptr->inputs[i].index;
      output_grads[in_node.get()][index].grads.emplace_back(input_grads[i]);
    }
  }

  // Aggregate gradients of each GradEntry
  for (auto& map_pair : output_grads) {
    for (GradEntry& grad_entry : map_pair.second) {
      // Aggregate sum if there haven't been.
      if (!grad_entry.sum.node) {
        grad_entry.sum = agg_fun(grad_entry.grads);
      }
    }
  }
  
  // Make the result graph.
  Graph ret;
  // Merge forward graph's outputs. Note that we don't keep the original attributes in
  // forward graph since they may be invalid due to the graph change.
  // TODO(minjie): Should have mechanism to declare which attribute is invalid after this
  // GradientPass.
  ret.outputs.insert(ret.outputs.end(), src.outputs.begin(), src.outputs.end());
  // Also put the xs' grads in the outputs.
  for (const NodeEntry& e : xs) {
    const GradEntry& entry = output_grads[e.node.get()][e.index];
    CHECK_NOTNULL(entry.sum.node);
    ret.outputs.push_back(entry.sum);
  }

  // Save the entry mapping to the "forward2backward" and "backward2forward" attributes.
  const IndexedGraph& idxgraph = ret.indexed_graph();
  std::unordered_map<uint32_t, std::vector<uint32_t>> forward2backward;
  std::unordered_map<uint32_t, std::vector<uint32_t>> backward2forward;
  for (const auto& map_pair : output_grads) {
    const Node* node = map_pair.first;  // Forward node.
    const uint32_t nodeid = idxgraph.node_id(node);
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const GradEntry& grad_ent = map_pair.second[i];
      if (!idxgraph.has_node(grad_ent.sum.node.get())) {
        // The node generating the gradient entry does not exist in the graph.
        // This means the gradient of this output entry is somehow not used in the backward
        // propagation. In this case, simply ignore it in the entry mapping.
        // NOTE: this also means not all the forward node entries are contained in the
        // gradient mapping.
        // LOG(INFO) << "Ignore forward node " << node->attrs.name << " output#" << i;
        continue;
      }
      const uint32_t forward_entid = idxgraph.entry_id(nodeid, i);
      // First map the forward entry to the output of the grad summation.
      const uint32_t gradsum_entid = idxgraph.entry_id(map_pair.second[i].sum);
      forward2backward[forward_entid].push_back(gradsum_entid);
      backward2forward[gradsum_entid].push_back(forward_entid);
      //LOG(INFO) << "Map between: " << forward_entid << " and " << gradsum_entid;
      // Then map the forward entry to all the inputs of the grad summation.
      for (const NodeEntry& gradsum_in_ent : map_pair.second[i].grads) {
        const uint32_t gradsum_in_entid = idxgraph.entry_id(gradsum_in_ent);
        forward2backward[forward_entid].push_back(gradsum_in_entid);
        backward2forward[gradsum_in_entid].push_back(forward_entid);
        //LOG(INFO) << "Map between: " << forward_entid << " and " << gradsum_in_entid;
      }
    }
  }
  ret.attrs["forward2backward"] = std::make_shared<any>(std::move(forward2backward));
  ret.attrs["backward2forward"] = std::make_shared<any>(std::move(backward2forward));

  return ret;
}

// register pass
NNVM_REGISTER_PASS(Gradient)
.describe("Return a gradient graph of src.attrs[\"ys\"] wrt src.attrs[\"xs\"]")
.set_body(Gradient)
.set_change_graph(true)
// Output NodeEntry of forward graph.
.depend_graph_attr("grad_ys")
// Input NodeEntries of forward graph; also gradient targets.
.depend_graph_attr("grad_xs")
// Output gradient NodeEntries of forward graph; also gradient targets.
.depend_graph_attr("grad_ys_out_grad")
// A map from forward entry to its corresponding backward entry.
.provide_graph_attr("forward2backward")
// A map from backward entry to its corresponding forward entry.
.provide_graph_attr("backward2forward")
;

}  // namespace
}  // namespace pass
}  // namespace nnvm
