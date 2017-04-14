/*!
 * Copyright (c) 2017 by Contributors
 * \file split_distributed_graph.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <queue>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

struct SplitGraphInputs {
  const Graph& src;
  const AddressVector& address_vec;
  const std::string localhost;
  const ShapeVector& shape_vec;
  const DTypeVector& dtype_vec;
  const Op* copy_op;
  const Op* net_init_op;
  const Op* net_send_op;
  const Op* net_recv_op;
  const Op* net_send_sink_op;
  const size_t num_forward_inputs;
  const size_t num_forward_outputs;
  const IndexedGraph& idx;
};

struct SplitGraphOutputs {
  Graph ret;
  OutputIdxMap output_idx_reverse_map;
  NodeEntry senders_sink;
  std::map<uint32_t, NodePtr> old_nid_to_new_node;
  std::map<Node*, uint32_t> new_node_to_old_nid;
  std::map<uint32_t, NodeEntry> old_eid_to_new_entry;
  std::map<std::pair<Node*, uint32_t>, uint32_t> new_entry_to_old_eid;
  std::map<uint32_t, NodeEntry> copy_entry_map;
  std::map<uint32_t, bool> copy_node_preserve;
  size_t num_new_forward_inputs;
  size_t num_removed_forward_inputs;
  size_t num_new_forward_outputs;
  size_t num_removed_forward_outputs;
  uint64_t tensor_id;
};

static bool SameNetAddress(const std::string& address1,
                           const std::string& address2) {
  return address1 == address2;
}

static NodeEntry CreateNetInit(SplitGraphInputs& in,
                               bool* already_has_init_node) {
  NodeEntry ret;
  NodePtr node = Node::Create();
  *already_has_init_node = false;
  for (uint32_t nid = 0; nid < in.idx.num_nodes(); ++nid) {
    if (in.idx[nid].source->op() == in.net_init_op) {
      CHECK(!(*already_has_init_node)) <<  "Have more than one p2pnet_init op.";
      *node = *(in.idx[nid].source);
      ret = NodeEntry{node, 0, 0};
      *already_has_init_node = true;
    }
  }
  if (!(*already_has_init_node)) {
    node->attrs.op = in.net_init_op;
    node->attrs.name = "NetInit";
    std::ostringstream os;
    os << in.localhost;
    node->attrs.dict["address"] = os.str();
    node->attrs.op->attr_parser(&(node->attrs));
    os << "_NetInit_redundant_var";
    node->inputs.emplace_back(Symbol::CreateVariable(os.str()).outputs[0]);
    ret = NodeEntry{std::move(node), 0, 0};
  }
  return ret;
}

// TODO: Currently I hardcode the parameters in CreateNetSendNode and
// CreateNetRecvNode. A cleaner way to create these parameters may be required.
static NodePtr CreateNetSendNode(const NodeEntry& init_node,
                                 const NodeEntry& data_node,
                                 const std::string& receiver_address,
                                 const std::string tensor_id,
                                 const Op* send_op) {
  NodePtr node = Node::Create();
  std::ostringstream os;
  os << receiver_address << "_" << tensor_id << "_sender" ;
  node->attrs.op = send_op;
  node->attrs.name = os.str();
  node->attrs.dict["tensor_id"] = tensor_id;
  node->attrs.dict["address"] = receiver_address;
  node->attrs.op->attr_parser(&(node->attrs));
  node->inputs.emplace_back(data_node);
  node->inputs.emplace_back(init_node);
  return node;
}

static NodePtr CreateNetRecvNode(const NodeEntry& init_node,
                                 const std::string& sender_address,
                                 const std::string tensor_id,
                                 const TShape& shape, const int dtype,
                                 const Op* recv_op) {
  NodePtr node = Node::Create();
  std::ostringstream os;
  os << sender_address << "_" << tensor_id << "_receiver";
  node->attrs.op = recv_op;
  node->attrs.name = os.str();
  node->attrs.dict["tensor_id"] = tensor_id;
  node->attrs.dict["address"] = sender_address;
  static const char* dtype_enum[] = {"float32", "float64", "float16", "uint8",
                                     "int32"};
  //os.str(""); os.clear(); os << dtype;
  node->attrs.dict["dtype"] = dtype_enum[dtype];
  os.str(""); os.clear(); os << shape;
  node->attrs.dict["shape"] = os.str();
  node->attrs.op->attr_parser(&(node->attrs));
  //os.str(""); os.clear();
  //os << sender_address << "_" << tensor_id << "_NetRecv_redundant_var";
  //node->inputs.emplace_back(Symbol::CreateVariable(os.str()).outputs[0]);
  node->inputs.emplace_back(init_node);
  return node;
}

static std::string CreateIdentity(const std::string& seed1,
                                  const std::string& seed2,
                                  const std::string& seed3,
                                  const std::string& seed4,
                                  const std::string& seed5) {
  static std::map<std::string, std::string[5]> id_map;
  std::cout << seed3 << seed1 << seed5 << std::endl;
  uint64_t id = std::stol(seed1 + seed3 + seed5);
  //size_t id =
    //((std::hash<std::string>{}(seed1) ^
      //std::hash<std::string>{}(seed2) ^
      //std::hash<std::string>{}(seed3) ^
      //std::hash<std::string>{}(seed4) ^
      //std::hash<std::string>{}(seed5)) << 1) % 100000000 + 90000000;
  std::string ret = std::to_string(id);
  if (id_map.count(ret) > 0) {
    std::cout << "Duplicated id : " << ret << std::endl
              << seed1 << " " << seed2 << " " << seed3 << " " << seed4 << " " << seed5
              << std::endl
              << id_map[ret][0] << " " << id_map[ret][1] << " " << id_map[ret][2] << " "
              << id_map[ret][3] << " " << id_map[ret][4]
              << std::endl;
    CHECK(id_map.count(ret) == 0);
  }
  id_map[ret][0] = seed1;
  id_map[ret][1] = seed2;
  id_map[ret][2] = seed3;
  id_map[ret][3] = seed4;
  id_map[ret][4] = seed5;
  return ret;
}

static void CheckAndSplitInputs(const struct SplitGraphInputs& in,
                                const uint32_t old_nid,
                                const NodePtr& new_node,
                                const NodeEntry& net_init_entry,
                                struct SplitGraphOutputs* out) {

  auto CreateInputEntry = [&in, &out]
      (NodePtr node, const NodePtr input_node, bool use_old_ientry_info,
       const IndexedGraph::NodeEntry& old_ientry) {
    const uint32_t index = (use_old_ientry_info) ? old_ientry.index : 0;
    const uint32_t version = (use_old_ientry_info) ? old_ientry.version : 0;
    const auto new_input_entry = NodeEntry{input_node, index, version};
    node->inputs.push_back(new_input_entry);
    const uint32_t old_eid = in.idx.entry_id(old_ientry);
    out->old_eid_to_new_entry[old_eid] = new_input_entry;
    out->new_entry_to_old_eid[std::make_pair(input_node.get(),
                                             new_input_entry.index)] = old_eid;
    return new_input_entry;
  };

  const auto& node_address = in.address_vec[old_nid];
  for (size_t i = 0; i < in.idx[old_nid].inputs.size(); i++) {
    const auto& input_ientry = in.idx[old_nid].inputs[i];
    const auto& input_address = in.address_vec[input_ientry.node_id];
    const auto& input_inode = in.idx[input_ientry.node_id];
    const auto& input_node = in.idx[input_ientry.node_id].source;
    if (input_node->op() != in.copy_op) {
      // Input must be on the same machine.
      CHECK(SameNetAddress(input_address, node_address) ||
            in.idx[old_nid].source->op() == in.copy_op)
          << input_node->attrs.name << "=" << input_address << ", "
          << new_node->attrs.name << "=" << node_address;
      if (SameNetAddress(node_address, in.localhost) &&
          SameNetAddress(input_address, in.localhost)) {
        CreateInputEntry(new_node,
                         out->old_nid_to_new_node.at(input_ientry.node_id),
                         true, input_ientry);
      }
      continue;
    } else if (input_node->inputs[0].node->op() == in.net_init_op) {
      // For a copy node whose input is a net_init_op, it's useless to us.
      continue;
    }

    const auto& sender_inode = input_inode.inputs[0];
    const auto& sender_old_nid = sender_inode.node_id;
    const auto& sender_address = in.address_vec[sender_old_nid];
    out->tensor_id += 1;
    //const std::string net_id =
      //CreateIdentity(std::to_string(sender_old_nid) + sender_address +
                     //std::to_string(i) +
                     //node_address + std::to_string(old_nid));
      //CreateIdentity(std::to_string(sender_old_nid), sender_address,
                     //std::to_string(i),
                     //node_address, std::to_string(old_nid));

    const auto& it = out->copy_entry_map.find(in.idx.entry_id(input_ientry));
    if (it != out->copy_entry_map.end()) {
      // No need to do anything if the corresponding op is sender.
      if (SameNetAddress(node_address, in.localhost)) {
        new_node->inputs.push_back(it->second);
      }
    } else if (SameNetAddress(node_address, in.localhost) &&
               SameNetAddress(sender_address, in.localhost)) {
      CHECK(in.idx[old_nid].source->op() != in.net_init_op &&
            in.idx[old_nid].source->op() != in.net_send_op &&
            in.idx[old_nid].source->op() != in.net_recv_op);
      CHECK(in.idx[sender_old_nid].source->op() != in.net_init_op &&
            in.idx[sender_old_nid].source->op() != in.net_send_op &&
            in.idx[sender_old_nid].source->op() != in.net_recv_op);
      CreateInputEntry(new_node,
                       out->old_nid_to_new_node.at(input_ientry.node_id),
                       true, input_ientry);
      out->copy_node_preserve[input_ientry.node_id] = true;
    } else if (SameNetAddress(node_address, in.localhost)) {
      if (in.idx[old_nid].source->op() != in.net_recv_op) {
        // Inserts a net_recv_op node to replace the copy_op node.
        auto recv_node =
            CreateNetRecvNode(net_init_entry, sender_address, 
                              std::to_string(out->tensor_id),
                              in.shape_vec[in.idx.entry_id(input_ientry)],
                              in.dtype_vec[in.idx.entry_id(input_ientry)],
                              in.net_recv_op);
        //out->num_new_forward_inputs++;
        // Replaces the copy node in all the maps. We have to remove the copy
        // node from new_node_to_old_nid. Otherwise, the legacy mapping may
        // cause some problems.
        auto new_input_entry =
            CreateInputEntry(new_node, recv_node, false, input_ientry);
        if (out->old_nid_to_new_node.count(input_ientry.node_id) >= 1) {
          auto copy_node = out->old_nid_to_new_node.at(input_ientry.node_id);
          out->new_node_to_old_nid.erase(copy_node.get());
        }
        out->old_nid_to_new_node[input_ientry.node_id] = recv_node;
        out->new_node_to_old_nid[recv_node.get()] = input_ientry.node_id;
        out->copy_entry_map[in.idx.entry_id(input_ientry)] = new_input_entry;

#if 1
        {
          std::vector<nnvm::NodeEntry> recv = {input_node->inputs[0]};
          NodePtr copy = nullptr;
	  DFSVisit(recv, [&copy, &in, &out, &recv_node] (const nnvm::NodePtr& n) {
             if (n->attrs.op == in.copy_op) {
               const auto& address = in.address_vec.at(in.idx.node_id(n->inputs[0].node.get()));
               if (address == in.localhost) {
               	 copy = n;
#if 0
            	 auto copy_entry = NodeEntry{copy, 0, 0};
            	 recv_node->control_deps.push_back(
        		out->copy_entry_map.at(in.idx.entry_id(copy_entry)).node);
#endif
	       }
             }
	  });
#if 1
          if (copy != nullptr) {
            auto copy_entry = NodeEntry{copy, 0, 0};
            recv_node->control_deps.push_back(
        	out->copy_entry_map.at(in.idx.entry_id(copy_entry)).node);
            std::cout << recv_node->attrs.name << " depends on " 
		      << out->copy_entry_map.at(in.idx.entry_id(copy_entry)).node->attrs.name
                       << std::endl;
          }
#endif
        }
#endif
      }
    } else if (SameNetAddress(sender_address, in.localhost)) {
      CHECK(!SameNetAddress(input_address, in.localhost));
      NodePtr sender_node =
          out->old_nid_to_new_node.at(
              in.idx.node_id(input_node->inputs[0].node.get()));
      if (in.idx[sender_old_nid].source->op() != in.net_send_op) {
        // Inserts a net_send_op node to replace the copy_op node.
        sender_node = CreateNetSendNode(
                          net_init_entry,
                          NodeEntry{out->old_nid_to_new_node.at(sender_old_nid),
                                    sender_inode.index,
                                    sender_inode.version},
                          node_address, std::to_string(out->tensor_id), 
                          in.net_send_op);
      }
      const auto sender_entry = NodeEntry{sender_node, 0, 0};
      //if (out->old_nid_to_new_node.count(input_ientry.node_id) >= 1) {
        //auto copy_node = out->old_nid_to_new_node.at(input_ientry.node_id);
        //out->new_node_to_old_nid.erase(copy_node.get());
        //out->old_nid_to_new_node[input_ientry.node_id] = sender_node;
        //out->new_node_to_old_nid[sender_node.get()] = input_ientry.node_id;
      //}
      out->copy_entry_map[in.idx.entry_id(input_ientry)] = sender_entry;
      out->senders_sink.node->control_deps.push_back(sender_node);
    } else {
      CHECK(!SameNetAddress(input_address, in.localhost));
    }
  }
}

static void RemoveUnusedCopyNode(const struct SplitGraphInputs& in,
                                 struct SplitGraphOutputs* out) {
  (void) in;
  for (const auto kv : out->copy_node_preserve) {
    if (!kv.second) {
      auto copy_node = out->old_nid_to_new_node.at(kv.first);
      out->new_node_to_old_nid.erase(copy_node.get());
      out->old_nid_to_new_node.erase(kv.first);
    }
  }
}

static void UpdateGraphAttributes(const struct SplitGraphInputs& in,
                                  struct SplitGraphOutputs* out) {
  std::cout << "SplitDistributedGraph is updating attributes." << std::endl;
  const auto& new_idx = out->ret.indexed_graph();
  ShapeVector new_shape_vec(new_idx.num_node_entries());
  DTypeVector new_dtype_vec(new_idx.num_node_entries());
  NodeIdMap node_id_map;
  EntryIdMap entry_id_map;
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const auto it = out->new_node_to_old_nid.find(
                        const_cast<Node*>(new_idx[nid].source));
    if (it != out->new_node_to_old_nid.end()) {
      node_id_map[nid] = it->second;
    }
    const size_t num_outputs = new_idx[nid].source->num_outputs();
    for (size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
      const size_t eid = new_idx.entry_id(nid, output_idx);
      if (it == out->new_node_to_old_nid.end()) {
        TShape shape(1);
        shape[0] = 1;
        new_shape_vec[eid] = shape;
        new_dtype_vec[eid] = 0;
      } else {
        const size_t old_eid = in.idx.entry_id(it->second, output_idx);
        new_shape_vec[eid] = in.shape_vec[old_eid];
        new_dtype_vec[eid] = in.dtype_vec[old_eid];
        entry_id_map[eid] = old_eid;
      }
    }
  }

  out->ret.attrs["context"] = in.src.attrs.at("context");
  out->ret.attrs["shape"] =
      std::make_shared<dmlc::any>(std::move(new_shape_vec));
  out->ret.attrs["dtype"] =
      std::make_shared<dmlc::any>(std::move(new_dtype_vec));
  out->ret.attrs["node_id_map"] =
      std::make_shared<dmlc::any>(std::move(node_id_map));
  out->ret.attrs["entry_id_map"] =
      std::make_shared<dmlc::any>(std::move(entry_id_map));
  out->ret.attrs["output_idx_reverse_map"] =
      std::make_shared<dmlc::any>(std::move(out->output_idx_reverse_map));
  out->ret.attrs["num_forward_inputs"] =
      std::make_shared<dmlc::any>(in.num_forward_inputs +
                                  out->num_new_forward_inputs -
                                  out->num_removed_forward_inputs);
  //std::cout << "num_forward_inputs : "
            //<< in.num_forward_inputs << " "
            //<< out->num_new_forward_inputs << " "
            //<< out->num_removed_forward_inputs << " "  << std::endl;
  out->ret.attrs["num_forward_outputs"] =
      std::make_shared<dmlc::any>(in.num_forward_outputs +
                                  out->num_new_forward_outputs -
                                  out->num_removed_forward_outputs);
  //std::cout << "num_forward_outputs : "
            //<< in.num_forward_outputs << " "
            //<< out->num_new_forward_outputs << " "
            //<< out->num_removed_forward_outputs << " "  << std::endl;
}

Graph SplitDistributedGraph(Graph src) {
  // Inputs
  struct SplitGraphInputs in = {
    src,
    src.GetAttr<AddressVector>("address"),
    src.GetAttr<std::string>("localhost"),
    src.GetAttr<ShapeVector>("shape"),
    src.GetAttr<DTypeVector>("dtype"),
    Op::Get(src.GetAttr<std::string>("device_copy_op")),
    Op::Get(src.GetAttr<std::string>("p2pnet_init_op")),
    Op::Get(src.GetAttr<std::string>("p2pnet_send_op")),
    Op::Get(src.GetAttr<std::string>("p2pnet_recv_op")),
    Op::Get("P2PNetSendSink"),
    src.GetAttr<size_t>("num_forward_inputs"),
    src.GetAttr<size_t>("num_forward_outputs"),
    src.indexed_graph(),
  };

  auto senders_sink = Node::Create();
  senders_sink->attrs.op = in.net_send_sink_op;
  senders_sink->attrs.name = "SendersSink";

  struct SplitGraphOutputs out = {
    Graph(),
    OutputIdxMap(),
    NodeEntry{senders_sink, 0, 0},
    std::map<uint32_t, NodePtr>(),
    std::map<Node*, uint32_t>(),
    std::map<uint32_t, NodeEntry>(),
    std::map<std::pair<Node*, uint32_t>, uint32_t>(),
    std::map<uint32_t, NodeEntry> (),
    std::map<uint32_t, bool> (),
    0,
    0,
    0,
    0,
    0
  };

  std::set<std::string> address_set(in.address_vec.begin(),
                                    in.address_vec.end());
  if (address_set.size() == 1) {
    out.ret.outputs = in.src.outputs;
    out.ret.attrs = in.src.attrs;
    return out.ret;
  }

  bool already_has_init_node = false;
  NodeEntry net_init_entry = CreateNetInit(in, &already_has_init_node);
  if (!already_has_init_node) {
    out.num_new_forward_inputs += 1;
  }

  // Finds the nodes for the local machine; removes copy-op and inserts the
  // corresponding p2psend or p2precv op for the local machine.
  std::cout << "SplitDistributedGraph pass begins." << std::endl;
  size_t num_input_encountered = 0;

  for (uint32_t old_nid = 0, old_output_idx = 0, new_output_idx = 0;
       old_nid < in.idx.num_nodes(); ++old_nid) {
    LOG(INFO) << "Node #" << old_nid << " "
      << in.idx[old_nid].source->attrs.name;
    // Always creates a new node for local nodes even if it may be discarded
    // later, e.g, a copy node between two machines. There is one exception!!
    // MXNet implicitly assumes that variables can not be changed.  As a result,
    // we should keep variables even if this makes the code inconsistent.
    const auto& node_address = in.address_vec[old_nid];
    NodePtr new_node = nullptr;

    if (SameNetAddress(node_address, in.localhost)) {
      if (in.idx[old_nid].source->is_variable()) {
        // FIXME: This is very hacky. Any better way to get NodePtr ?
        for (uint32_t nid = 0; nid < in.idx.num_nodes(); ++nid) {
          for (uint32_t control_idx = 0;
               control_idx < in.idx[nid].control_deps.size(); ++control_idx) {
            if (in.idx[nid].control_deps[control_idx] == old_nid) {
              new_node = in.idx[nid].source->control_deps[control_idx];
              break;
            }
          }
          if (new_node != nullptr) {
            break;
          }
          for (uint32_t input_idx = 0;
               input_idx < in.idx[nid].inputs.size(); ++input_idx) {
            if (in.idx[nid].inputs[input_idx].node_id == old_nid) {
              new_node = in.idx[nid].source->inputs[input_idx].node;
              break;
            }
          }
        }
        CHECK(new_node != nullptr);
        // Very hacky. Every variables must depend on NetInit node.
        // Otherwise, it is possible that NetInit_redundant_var will be placed
        // to "backward" input.
        new_node->control_deps.push_back(net_init_entry.node);
      } else {
        new_node = Node::Create();
        new_node->attrs = in.idx[old_nid].source->attrs;
      }
      for (size_t i = 0; i < in.idx[old_nid].control_deps.size(); ++i) {
        uint32_t old_depend_nid = in.idx[old_nid].control_deps[i];
        if (SameNetAddress(in.address_vec[old_depend_nid], in.localhost)) {
          new_node->control_deps.push_back(
              out.old_nid_to_new_node.at(old_depend_nid));
        }
      }
      out.old_nid_to_new_node[old_nid] = new_node;
      out.new_node_to_old_nid[new_node.get()] = old_nid;
      if (new_node->attrs.op == in.copy_op) {
        out.copy_node_preserve[old_nid] = false;
      }
    }
    if (in.idx[old_nid].source->is_variable()) {
      num_input_encountered++;
      if (num_input_encountered <= in.num_forward_inputs &&
          !SameNetAddress(node_address, in.localhost)) {
        out.num_removed_forward_inputs++;
      }
    }

    CheckAndSplitInputs(in, old_nid, new_node, net_init_entry, &out);

    const auto& output_entry = in.idx.outputs()[old_output_idx];
    if (output_entry.node_id == old_nid) {
      if (SameNetAddress(in.address_vec[output_entry.node_id], in.localhost)) {
        const auto new_output_entry =
            NodeEntry{new_node, output_entry.index, output_entry.version};
        const uint32_t old_eid = in.idx.entry_id(output_entry);
        out.old_eid_to_new_entry[old_eid] = new_output_entry;
        out.new_entry_to_old_eid[std::make_pair(new_node.get(),
                                                new_output_entry.index)]
            = old_eid;
        out.ret.outputs.emplace_back(new_output_entry);
        out.output_idx_reverse_map[old_output_idx] = new_output_idx;
        new_output_idx++;
      } else {
        if (old_output_idx + 1 <= in.num_forward_outputs) {
          out.num_removed_forward_outputs++;
        }
      }
      if (old_output_idx + 1 == in.num_forward_outputs) {
        out.ret.outputs.emplace_back(out.senders_sink);
        new_output_idx++;
        out.num_new_forward_outputs++;
      }
      old_output_idx++;
    }
  }

  for (auto it = out.ret.outputs.begin(); it < out.ret.outputs.end(); it++) {
    if (it->node->op() == in.net_send_sink_op) {
      if (it->node->control_deps.size() == 0) {
        out.ret.outputs.erase(it);
        out.num_new_forward_outputs--;
      }
      break;
    }
  }

  //RemoveUnusedCopyNode(in, &out);
  UpdateGraphAttributes(in, &out);
  std::cout << "SplitDistributedGraph pass finished." << std::endl;

  const auto& retidx = out.ret.indexed_graph();
  std::cout << "digraph {" << std::endl;
  std::vector<bool> touched_nodes(retidx.num_nodes(), false);
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    const auto& n = retidx[nid];
    if (n.source->attrs.name == "SendersSink") {
      continue;
    }
    for (const auto& in : n.inputs) {
      if (n.source->attrs.name == "NetInit" || retidx[in.node_id].source->attrs.name == "NetInit") {
        continue;
      }
      std::cout << "\tn" << in.node_id << " -> n" << nid << std::endl;
      touched_nodes[nid] = true;
      touched_nodes[in.node_id] = true;
    }
    for (uint32_t control_nid : n.control_deps) {
      if (n.source->attrs.name == "NetInit" || retidx[control_nid].source->attrs.name == "NetInit") {
        continue;
      }
      std::cout << "\tn" << control_nid << " -> n" << nid
        << " [style=dotted]" << std::endl;
      touched_nodes[control_nid] = true;
      touched_nodes[nid] = true;
    }
  }
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    if (!touched_nodes[nid]) {
      continue;
    }
    std::cout << "\t\tn" << nid << " [label=\""
      << "n" << nid << "_" << retidx[nid].source->attrs.name << "\"]" << std::endl;
  }
  std::cout << "}" << std::endl;

  return out.ret;
}

NNVM_REGISTER_PASS(SplitDistributedGraph)
.describe("Split the origin graph into several subgraphs accross different "
          "machines. This pass should be done AFTER PlaceDevice pass because "
          "it needs the information of which node should be placed on which "
          "machine.")
.set_body(SplitDistributedGraph)
.set_change_graph(true)
.provide_graph_attr("shape")
.provide_graph_attr("dtype")
.provide_graph_attr("node_id_map")
.provide_graph_attr("entry_id_map")
.provide_graph_attr("output_idx_reverse_map")
.provide_graph_attr("num_forward_inputs")
.provide_graph_attr("num_forward_outputs")
.depend_graph_attr("address")
.depend_graph_attr("localhost")
.depend_graph_attr("context")
.depend_graph_attr("shape")
.depend_graph_attr("dtype")
.depend_graph_attr("device_copy_op")
.depend_graph_attr("p2pnet_init_op")
.depend_graph_attr("p2pnet_send_op")
.depend_graph_attr("p2pnet_recv_op")
.depend_graph_attr("num_forward_inputs")
.depend_graph_attr("num_forward_outputs");
}  // namespace
}  // namespace pass
}  // namespace nnvm
