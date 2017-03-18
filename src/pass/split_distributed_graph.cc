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

static bool SameNetAddress(const std::string& address1,
                           const std::string& address2) {
  return address1 == address2;
}

static NodeEntry CreateNetInit(Graph& graph, const std::string localhost,
                               const Op* init_op, bool* already_has_init_node) {
  NodeEntry ret;
  NodePtr node = Node::Create();
  *already_has_init_node = false;
  const auto& idx = graph.indexed_graph();
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (idx[nid].source->op() == init_op) {
      CHECK(!(*already_has_init_node)) <<  "Have more than one p2pnet_init op.";
      *node = *idx[nid].source;
      ret = NodeEntry{node, 0, 0};
      *already_has_init_node = true;
    }
  }
  if (!(*already_has_init_node)) {
    node->attrs.op = init_op;
    node->attrs.name = "NetInit";
    std::ostringstream os;
    os << localhost;
    node->attrs.dict["address"] = os.str();
    node->attrs.op->attr_parser(&(node->attrs));
    os << "_NetInit_redudent_var";
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
  os.str(""); os.clear();
  os << sender_address << "_" << tensor_id << "_NetRecv_redudent_var";
  node->inputs.emplace_back(Symbol::CreateVariable(os.str()).outputs[0]);
  node->inputs.emplace_back(init_node);
  return node;
}

static std::string CreateIdentity(const std::string& seed) {
  size_t id = (std::hash<std::string>{}(seed)) % 10000000 + 90000000;
  return std::to_string(id);
}

Graph SplitDistributedGraph(Graph src) {
  const auto& address_vec = src.GetAttr<AddressVector>("address");
  const std::string localhost = src.GetAttr<std::string>("localhost");
  const auto& shape_vec = src.GetAttr<ShapeVector>("shape");
  const auto& dtype_vec = src.GetAttr<DTypeVector>("dtype");
  const Op* copy_op = Op::Get(src.GetAttr<std::string>("device_copy_op"));
  const Op* net_init_op = Op::Get(src.GetAttr<std::string>("p2pnet_init_op"));
  const Op* net_send_op = Op::Get(src.GetAttr<std::string>("p2pnet_send_op"));
  const Op* net_recv_op = Op::Get(src.GetAttr<std::string>("p2pnet_recv_op"));
  const size_t num_forward_inputs = src.GetAttr<size_t>("num_forward_inputs");
  const size_t num_forward_outputs = src.GetAttr<size_t>("num_forward_outputs");
  const IndexedGraph& idx = src.indexed_graph();
  std::vector<NodeEntry> new_outputs;
  std::map<uint32_t, NodePtr> old_nid_to_new_node;
  std::map<Node*, uint32_t> new_node_to_old_nid;
  std::map<uint32_t, NodeEntry> old_eid_to_new_entry;
  std::map<std::pair<Node*, uint32_t>, uint32_t> new_entry_to_old_eid;
  std::map<uint32_t, NodeEntry> copy_op_map;
  size_t num_new_forward_inputs = 0;
  size_t num_removed_forward_inputs = 0;
  size_t num_new_forward_outputs = 0;
  size_t num_removed_forward_outputs = 0;

  bool already_has_init_node = false;
  NodeEntry net_init_entry = CreateNetInit(src, localhost, net_init_op,
                                           &already_has_init_node);
  if (!already_has_init_node) {
    num_new_forward_inputs += 1;
  }

  // Finds the nodes for the local machine; removes copy-op and inserts the
  // corresponding p2psend or p2precv op for the local machine.
  std::cout << "SplitDistributedGraph pass begins." << std::endl;
  size_t num_input_encountered = 0;

  auto CreateInputEntry = [&idx, &old_eid_to_new_entry, &new_entry_to_old_eid]
      (NodePtr node, const NodePtr input_node, bool use_old_ientry_info,
       const IndexedGraph::NodeEntry& old_ientry) {
    const uint32_t index = (use_old_ientry_info) ? old_ientry.index : 0;
    const uint32_t version = (use_old_ientry_info) ? old_ientry.version : 0;
    const auto new_input_entry = NodeEntry{input_node, index, version};
    node->inputs.push_back(new_input_entry);
    const uint32_t old_eid = idx.entry_id(old_ientry);
    old_eid_to_new_entry[old_eid] = new_input_entry;
    new_entry_to_old_eid[std::make_pair(input_node.get(),
                                        new_input_entry.index)] = old_eid;
    return new_input_entry;
  };
  for (uint32_t old_nid = 0; old_nid < idx.num_nodes(); ++old_nid) {
    // Always creates a new node for local nodes even if it may be discarded
    // later, e.g, a copy node between two machines. There is one exception!!
    // MXNet implicitly assumes that variables can not be changed.  As a result,
    // we should keep variables even if this makes the code inconsistent.
    const auto& node_address = address_vec[old_nid];
    NodePtr new_node = nullptr;
    bool is_backward_input = false;
    if (idx[old_nid].source->is_variable()) {
      num_input_encountered++;
      if (num_input_encountered > num_forward_inputs) {
        is_backward_input = true;
      }
    }
    if (SameNetAddress(node_address, localhost)) {
      if (!idx[old_nid].source->is_variable()) {
        new_node = Node::Create();
        new_node->attrs = idx[old_nid].source->attrs;
        new_node->control_deps.reserve(idx[old_nid].control_deps.size());
        for (size_t i = 0; i < idx[old_nid].control_deps.size(); ++i) {
          uint32_t old_cid = idx[old_nid].control_deps[i];
          new_node->control_deps.push_back(old_nid_to_new_node[old_cid]);
        }
        old_nid_to_new_node[old_nid] = new_node;
        new_node_to_old_nid[new_node.get()] = old_nid;
      }
    } else if (is_backward_input) {
      num_removed_forward_inputs++;
    }

    // Check all inputs to see if we need insert send/receive.
    for (size_t i = 0; i < idx[old_nid].inputs.size(); i++) {
      const auto& input_ientry = idx[old_nid].inputs[i];
      const auto& input_address = address_vec[input_ientry.node_id];
      const auto& input_inode = idx[input_ientry.node_id];
      const auto& input_node = idx[input_ientry.node_id].source;
      if (input_node->op() != copy_op) {
        // Input must be on the same machine.
        CHECK(SameNetAddress(input_address, node_address) ||
              idx[old_nid].source->op() == copy_op)
            << input_node->attrs.name << "=" << input_address << ", "
            << new_node->attrs.name << "=" << node_address;
        if (SameNetAddress(node_address, localhost)) {
          if (input_node->is_variable()) {
            // Note: Must preserve the original variable. See the comment about
            // creating new nodes.
            NodePtr variable_node = idx[old_nid].source->inputs[i].node;
            old_nid_to_new_node[input_ientry.node_id] = variable_node;
            new_node_to_old_nid[variable_node.get()] = input_ientry.node_id;
          }
          CreateInputEntry(new_node, old_nid_to_new_node[input_ientry.node_id],
                           true, input_ientry);
        }
        continue;
      } else if (input_node->inputs[0].node->op() == net_init_op) {
        // For a copy node whose input is a net_init_op, it's useless to us.
        continue;
      }

      const auto& sender_inode = input_inode.inputs[0];
      const auto& sender_old_nid = sender_inode.node_id;
      const auto& sender_address = address_vec[sender_old_nid];
      const std::string net_id = CreateIdentity(input_node->attrs.name +
                                                sender_address);
      const auto& it = copy_op_map.find(idx.entry_id(input_ientry));
      if (it != copy_op_map.end()) {
        // No need to do anything if the corresponding op is sender.
        if (SameNetAddress(node_address, localhost)) {
          new_node->inputs.push_back(it->second);
        }
      } else if (SameNetAddress(node_address, localhost) &&
                 SameNetAddress(sender_address, localhost)) {
        CHECK(idx[old_nid].source->op() != net_init_op &&
              idx[old_nid].source->op() != net_send_op &&
              idx[old_nid].source->op() != net_recv_op);
        CHECK(idx[sender_old_nid].source->op() != net_init_op &&
              idx[sender_old_nid].source->op() != net_send_op &&
              idx[sender_old_nid].source->op() != net_recv_op);
        CreateInputEntry(new_node, old_nid_to_new_node[input_ientry.node_id],
                         true, input_ientry);
      } else if (SameNetAddress(node_address, localhost)) {
        if (idx[old_nid].source->op() != net_recv_op) {
          // Inserts a net_recv_op node to replace the copy_op node.
          auto recv_node =
              CreateNetRecvNode(net_init_entry, sender_address, net_id,
                                shape_vec[idx.entry_id(input_ientry)],
                                dtype_vec[idx.entry_id(input_ientry)],
                                net_recv_op);
          num_new_forward_inputs++;
          // Replaces the copy node in all the maps. We have to remove the copy
          // node from new_node_to_old_nid. Otherwise, the legacy mapping may
          // cause some problems.
          auto new_input_entry =
              CreateInputEntry(new_node, recv_node, false, input_ientry);
          auto copy_node = old_nid_to_new_node[input_ientry.node_id];
          new_node_to_old_nid.erase(copy_node.get());
          old_nid_to_new_node[input_ientry.node_id] = recv_node;
          new_node_to_old_nid[recv_node.get()] = input_ientry.node_id;
          copy_op_map[idx.entry_id(input_ientry)] = new_input_entry;
          std::cout << "Copy node address is : "
                    << address_vec[input_ientry.node_id] << std::endl;
          std::cout << "Receiver node address is : "
                    << node_address << std::endl;
        }
      } else if (SameNetAddress(sender_address, localhost)) {
        CHECK(!SameNetAddress(input_address, localhost));
        NodePtr sender_node =
            old_nid_to_new_node[idx.node_id(input_node->inputs[0].node.get())];
        if (idx[sender_old_nid].source->op() != net_send_op) {
          // Inserts a net_send_op node to replace the copy_op node.
          sender_node = CreateNetSendNode(
                            net_init_entry,
                            NodeEntry{old_nid_to_new_node[sender_old_nid],
                                       sender_inode.index,
                                       sender_inode.version},
                            node_address, net_id, net_send_op);
          // Replaces the copy node in all the maps.
          //old_nid_to_new_node[input_ientry.node_id] = node;
          //new_node_to_old_nid[node.get()] = input_ientry.node_id;
          std::cout << "Copy node address is : "
                    << address_vec[input_ientry.node_id] << std::endl;
          std::cout << "Sender node address is : "
                    << sender_address << std::endl;
        }
        const auto sender_entry = NodeEntry{sender_node, 0, 0};
        //const auto old_eid = idx.entry_id(input_inode.inputs[0]);
        //old_eid_to_new_entry[old_eid] = sender_entry;
        //new_entry_to_old_eid[std::make_pair(sender_entry.node.get(),
                                            //sender_entry.index)] = old_eid;
        copy_op_map[idx.entry_id(input_ientry)] = sender_entry;
        new_outputs.push_back(sender_entry);
        num_new_forward_outputs += 1;
      } else {
        CHECK(!SameNetAddress(input_address, localhost));
      }
    }
  }

  // Uses the send/recv nodes and the new nodes to make the new Graph.
  Graph ret;
  OutputIdxMap output_idx_reverse_map;
  uint32_t old_output_idx = 0;
  uint32_t new_output_idx = 0;
  for (const NodeEntry& e : src.outputs) {
    const uint32_t node_id = idx.node_id(e.node.get());
    if (old_nid_to_new_node.find(node_id) != old_nid_to_new_node.end()) {
      const auto& new_node = old_nid_to_new_node.at(node_id);
      const auto new_entry = NodeEntry{new_node, e.index, e.version};
      const uint32_t old_eid = idx.entry_id(e);
      old_eid_to_new_entry[old_eid] = new_entry;
      new_entry_to_old_eid[std::make_pair(new_node.get(), new_entry.index)]
          = old_eid;
      ret.outputs.emplace_back(new_entry);
      output_idx_reverse_map[old_output_idx] = new_output_idx;
      new_output_idx++;

    } else if (old_output_idx < num_forward_outputs) {
      num_removed_forward_outputs++;
    }
    old_output_idx++;
  }
  for (const auto& n : new_outputs) {
    // TODO: We currently assumes that all new outputs (creating from send
    // nodes) belong to forward output nodes. Needs to double check if this
    // assumption is correct.
    ret.outputs.emplace(ret.outputs.begin(), n);
  }

  std::cout << "SplitDistributedGraph is updating attributes." << std::endl;
  const auto& new_idx = ret.indexed_graph();
  ShapeVector new_shape_vec(new_idx.num_node_entries());
  DTypeVector new_dtype_vec(new_idx.num_node_entries());
  NodeIdMap node_id_map;
  EntryIdMap entry_id_map;
  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    const auto it = new_node_to_old_nid.find(
                        const_cast<Node*>(new_idx[nid].source));
    if (it != new_node_to_old_nid.end()) {
      node_id_map[nid] = it->second;
    }
    const size_t num_outputs = new_idx[nid].source->num_outputs();
    for (size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
      const size_t eid = new_idx.entry_id(nid, output_idx);
      if (it == new_node_to_old_nid.end()) {
        TShape shape(1);
        shape[0] = 1;
        new_shape_vec[eid] = shape;
        new_dtype_vec[eid] = 0;
      } else {
        const size_t old_eid = idx.entry_id(it->second, output_idx);
        new_shape_vec[eid] = shape_vec[old_eid];
        new_dtype_vec[eid] = dtype_vec[old_eid];
        entry_id_map[eid] = old_eid;
      }
    }
  }

  ret.attrs["context"] = src.attrs["context"];
  ret.attrs["shape"] = std::make_shared<dmlc::any>(std::move(new_shape_vec));
  ret.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(new_dtype_vec));
  ret.attrs["node_id_map"] =
      std::make_shared<dmlc::any>(std::move(node_id_map));
  ret.attrs["entry_id_map"] =
      std::make_shared<dmlc::any>(std::move(entry_id_map));
  ret.attrs["output_idx_reverse_map"] =
      std::make_shared<dmlc::any>(std::move(output_idx_reverse_map));
  ret.attrs["num_forward_inputs"] =
      std::make_shared<dmlc::any>(num_forward_inputs + num_new_forward_inputs -
                                  num_removed_forward_inputs);
  ret.attrs["num_forward_outputs"] =
      std::make_shared<dmlc::any>(num_forward_outputs +
                                  num_new_forward_outputs -
                                  num_removed_forward_outputs);
  std::cout << "SplitDistributedGraph pass finished." << std::endl;
  return ret;
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
// TODO: What's this for?
//DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);
}  // namespace
}  // namespace pass
}  // namespace nnvm
