/*!
 * Copyright (c) 2017 by Contributors
 * \file split_distributed_graph.cc
 * \brief
 * \author Chien-Chin Huang
*/
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

bool IsLocal(std::string address, std::string localhost) {
  std::size_t found = address.find(localhost);
  return found != std::string::npos;
}

// TODO: Determine this port dynamically.
static unsigned port = 9000;
const std::string localhost = "127.0.0.1";

void CreateNetInitNode(NodePtr node, const Op* init_op) {
  node->attrs.op = init_op;
  node->attrs.name = "NetInit";
  std::ostringstream os;
  os << "127.0.0.1:" << port;
  node->attrs.dict["address"] = os.str();
  node->attrs.op->attr_parser(&(node->attrs));
  // TODO: Create a variable as the input.
  //node->inputs.emplace_back(NodeEntry{init_node, 0, 0});
}

// TODO: Currently I hardcode the parameters in CreateNetSendNode and 
// CreateNetRecvNode. A cleaner way to create these parameters may be required.
NodePtr CreateNetSendNode(const NodeEntry& init_node, 
                          const NodeEntry& data_node, 
                          const std::string& receiver_address,
                          const std::string tensor_id, const Op* send_op) {
  NodePtr node = Node::Create();
  std::ostringstream os;
  os << receiver_address << "_" << tensor_id << "_sender" ;
  node->attrs.op = send_op;
  node->attrs.name = os.str();
  node->attrs.dict["tensor_id"] = tensor_id;
  node->attrs.dict["address"] = receiver_address;
  node->attrs.op->attr_parser(&(node->attrs));
  std::cout << "CreateNetSendNode " << init_node.node->attrs.name << std::endl;
  std::cout << "CreateNetSendNode " << data_node.node->attrs.name << std::endl;
  node->inputs.emplace_back(data_node);
  node->inputs.emplace_back(init_node);
  return node;
}

NodePtr CreateNetRecvNode(const NodeEntry& init_node,
                          const std::string& sender_address,
                          const std::string tensor_id, const TShape& shape,
                          const int dtype, const Op* recv_op) {
  NodePtr node = Node::Create();
  std::ostringstream os;
  os << sender_address << "_" << tensor_id << "_receiver";
  node->attrs.op = recv_op;
  node->attrs.name = os.str();
  node->attrs.dict["tensor_id"] = tensor_id;
  node->attrs.dict["address"] = sender_address;
  os.clear(); os << dtype;
  node->attrs.dict["dtype"] = os.str();
  os.clear(); os << shape;
  node->attrs.dict["shape"] = os.str();
  node->attrs.op->attr_parser(&(node->attrs));
  /* TODO: Create a variable */
  os << "_redudent_var";
  node->inputs.emplace_back(Symbol::CreateVariable(os.str()).outputs[0]);
  node->inputs.emplace_back(init_node);
  return node;
}

static std::string CreateIdentity(const std::string& seed) {
  size_t id = (std::hash<std::string>{}(seed)) % 10000000 + 90000000;
  return std::to_string(id);
}

Graph SplitDistributedGraph(Graph src) {
  //CHECK(src.attrs.count("device"))
      //<< "Need graph attribute \"device_group_attr_key\" in PlaceDevice";

  const auto& address_vec = src.GetAttr<AddressVector>("address");
  const auto& shape_vec = src.GetAttr<ShapeVector>("shape");
  const auto& dtype_vec = src.GetAttr<DTypeVector>("dtype");
  const Op* copy_op = Op::Get(src.GetAttr<std::string>("device_copy_op"));
  const Op* net_init_op = Op::Get(src.GetAttr<std::string>("p2pnet_init_op"));
  const Op* net_send_op = Op::Get(src.GetAttr<std::string>("p2pnet_send_op"));
  const Op* net_recv_op = Op::Get(src.GetAttr<std::string>("p2pnet_recv_op"));
  const IndexedGraph& idx = src.indexed_graph();

  // TODO: Get localhost from either net_init or "address".

  std::vector<NodeEntry> new_nodes;
  std::vector<int> discarded_local_nodes;
  // See if there is p2pnet_init op in the graph. If so, just reuses it.
  // Otherwse, creates a new one.
  NodePtr net_init_node = Node::Create();
  bool has_init_node = false;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    if (idx[nid].source->op() == net_init_op) {
      CHECK(!has_init_node) <<  "Have more than one p2pnet_init op.";
      *net_init_node = *idx[nid].source;
      has_init_node = true;
    }
  }
  if (!has_init_node) {
    CreateNetInitNode(net_init_node, net_init_op);
  }
  new_nodes.push_back(NodeEntry{std::move(net_init_node), 0, 0});
  
  // Finds the nodes for the local machine; removes copy-op and inserts the 
  // corresponding p2psend or p2precv op for the local machine.
  std::cout << "Begin" << std::endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& receiver_inode = idx[nid];
    if (receiver_inode.source->is_variable()) {
      std::cout << "is_variable" << std::endl;
      continue;
    }
    std::cout << "SplitDistributedGraph is handling with op = " 
              << receiver_inode.source->attrs.name << std::endl;

    for (size_t i = 0; i < receiver_inode.inputs.size(); ++i) {
      const auto& copy_inode = idx[receiver_inode.inputs[i].node_id];
      if (copy_inode.source->op() != copy_op ||
          idx[copy_inode.inputs[0].node_id].source->op() == net_init_op) {
        continue;
      }
      const auto& sender_inode = idx[copy_inode.inputs[0].node_id];
      const auto& receiver_address = 
          address_vec[idx.node_id(receiver_inode.source)];
      const auto& sender_address = 
          address_vec[idx.node_id(sender_inode.source)];
      const auto& copy_address = 
          address_vec[idx.node_id(copy_inode.source)];
      if (IsLocal(receiver_address, localhost) && 
          IsLocal(sender_address, localhost)) {
        // Keeps receiver, sender and copy nodes.
        CHECK(receiver_inode.source->op() != net_recv_op);
        CHECK(sender_inode.source->op() != net_send_op);
      } else if (IsLocal(receiver_address, localhost)) {
        if (IsLocal(copy_address, localhost)) {
          discarded_local_nodes.push_back(idx.node_id(copy_inode.source));
        }
        if (receiver_inode.source->op() != net_recv_op) {
          // If it's a net_recv_op node, just reuses it. Otherwise, creates one.
          const int eid = idx.entry_id(idx.node_id(sender_inode.source), 0);
          const std::string id = CreateIdentity(copy_inode.source->op()->name +
                                                sender_address);
          auto node = CreateNetRecvNode(new_nodes[0], sender_address, id, 
                                        shape_vec[eid], dtype_vec[eid],
                                        net_recv_op);
          new_nodes.push_back(NodeEntry{std::move(node), 0, 0});
        }
      } else if (IsLocal(sender_address, localhost)) {
        if (IsLocal(copy_address, localhost)) {
          discarded_local_nodes.push_back(idx.node_id(copy_inode.source));
        }
        if (sender_inode.source->op() != net_send_op) {
          // If it's a net_send_op node, just reuses it. Otherwise, creates one.
          // TODO: Creates a sender node and makes its input as sender_inode.
          //CreateNetSendNode(NodePtr init_node, const std::string& receiver_address,
                          //const std::string tensor_id, const Op* send_op) {
          const std::string id = CreateIdentity(copy_inode.source->op()->name +
                                                sender_address);
          auto node = CreateNetSendNode(new_nodes[0],
                                        copy_inode.source->inputs[0],
                                        receiver_address, id, net_send_op);
          new_nodes.push_back(NodeEntry{std::move(node), 0, 0});
        }
      } else {
        CHECK(!IsLocal(copy_address, localhost));
      }
    }
  }

  std::cout << "Finished" << std::endl;
  // Uses the new send/recv nodes and the preserved nodes to make the new Graph.
  Graph ret;
  for (const NodeEntry& e : src.outputs) {
    int node_id = idx.node_id(e.node.get());
    std::cout << address_vec[node_id] << " " << node_id << std::endl;
    std::cout << idx[node_id].source->attrs.name << std::endl;
    if (IsLocal(address_vec[node_id], localhost)) {
      if (discarded_local_nodes[node_id]) {
        continue;
      }
      std::cout << idx[node_id].source->attrs.name << std::endl;
      ret.outputs.emplace_back(e);
    }
  }
  if (has_init_node) {
    std::cout << "ERASE " << new_nodes[0].node->attrs.name << std::endl;
    std::cout << "ERASE " << new_nodes[1].node->attrs.name << std::endl;
    new_nodes.erase(new_nodes.begin());
    std::cout << "ERASE " << new_nodes[0].node->attrs.name << std::endl;
  } else {
    std::cout << "No ERASE" << std::endl;
  }
  for (const auto& n : new_nodes) {
    ret.outputs.emplace_back(n);
  }
  //ShapeVector new_shape_vec;
  //DTypeVector new_dtype_vec;
  //StorageVector new_storage_vec;
  const auto& new_idx = ret.indexed_graph();
  //std::cout << "-------------------@@" << std::endl;
  //for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    //std::cout << "-------------------A" << nid << " " << new_idx[nid].source->attrs.name << " " << (size_t)(new_idx[nid].source) << std::endl;
    //const uint32_t old_nid = idx.node_id(new_idx[nid].source);
    //std::cout << "-------------------A" << nid << " " << new_idx[nid].source->attrs.name << std::endl;
    //const uint32_t old_eid = idx.entry_id(old_nid, 0);
    //std::cout << "-------------------B" << old_eid << std::endl;
    //new_shape_vec.push_back(shape_vec[old_eid]);
    //new_dtype_vec.push_back(dtype_vec[old_eid]);
  //}
  //std::cout << "-------------------" << std::endl;
  //ret.attrs["shape"] = std::make_shared<dmlc::any>(std::move(new_shape_vec));
  //ret.attrs["dtype"] = std::make_shared<dmlc::any>(std::move(new_dtype_vec));

  for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
    std::cout << "Ret, node_id = " << nid << " name = " << new_idx[nid].source->attrs.name << std::endl;
  }
  for (const auto& node_entry : new_idx.outputs()) {
    std::cout << "Ret, node_entry node id = " << node_entry.node_id << std::endl;
  }
  for (const auto& kv : ret.attrs) {
    std::cout << "Ret, key of attr = " << kv.first << std::endl;
  }
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
.depend_graph_attr("address")
.depend_graph_attr("shape")
.depend_graph_attr("dtype")
.depend_graph_attr("device_copy_op")
.depend_graph_attr("p2pnet_init_op")
.depend_graph_attr("p2pnet_send_op")
.depend_graph_attr("p2pnet_recv_op");

// TODO: What's this for?
//DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);

}  // namespace
}  // namespace pass
}  // namespace nnvm
