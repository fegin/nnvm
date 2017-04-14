/*!
 *  Copyright (c) 2016 by Contributors
 * \file place_device.cc
 * \brief Inference the device of each operator given known information.
 *  Insert a copy node automatically when there is a cross device.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

template<typename AttrType>
void RemapEntryAttributes(
    Graph origin, Graph* newgraph,
    const vector<NodePtr>& new_node_map,
    const string& attr_name) {
  const IndexedGraph& origin_idx = origin.indexed_graph();
  const IndexedGraph& new_idx = newgraph->indexed_graph();
  CHECK_NE(origin.attrs.count(attr_name), 0);
  const vector<AttrType>& origin_attr_vec = origin.GetAttr<vector<AttrType>>(attr_name);
  CHECK_EQ(origin_attr_vec.size(), origin_idx.num_node_entries());
  vector<AttrType> new_attr_vec(new_idx.num_node_entries());
  vector<bool> visited_new_nodes(new_idx.num_nodes(), false);
  for (uint32_t origin_nid = 0; origin_nid < origin_idx.num_nodes(); ++origin_nid) {
    const Node* origin_node = origin_idx[origin_nid].source;
    const Node* new_node = new_node_map[origin_nid]?
      new_node_map[origin_nid].get() : origin_node;
    const uint32_t new_nid = new_idx.node_id(new_node);
    for (uint32_t i = 0; i < origin_node->num_outputs(); ++i) {
      const uint32_t origin_entid = origin_idx.entry_id(origin_nid, i);
      const uint32_t new_entid = new_idx.entry_id(new_nid, i);
      new_attr_vec[new_entid] = origin_attr_vec[origin_entid];
    }
    visited_new_nodes[new_nid] = true;
  }
  // Infer attributes for copy node.
  // TODO(minjie): this part of logic is not generic!!! We know this remaining
  // nodes are copy nodes.
  for (uint32_t nodeid = 0; nodeid < new_idx.num_nodes(); ++nodeid) {
    if (!visited_new_nodes[nodeid]) {
      const IndexedGraph::Node& inode = new_idx[nodeid];
      CHECK_EQ(inode.inputs.size(), 1);
      CHECK_EQ(inode.source->num_outputs(), 1);
      CHECK(visited_new_nodes[inode.inputs[0].node_id]);
      const uint32_t entid = new_idx.entry_id(nodeid, 0);
      const uint32_t inentid = new_idx.entry_id(inode.inputs[0]);
      new_attr_vec[entid] = new_attr_vec[inentid];
    }
  }
  newgraph->attrs[attr_name] = std::make_shared<any>(std::move(new_attr_vec));
}

// simply logic to place device according to device_group hint
// insert copy node when there is
Graph PlaceDevice(Graph src) {
  CHECK_NE(src.attrs.count("device_group_attr_key"), 0)
      << "Need graph attribute \"device_group_attr_key\" in PlaceDevice";
  CHECK_NE(src.attrs.count("device_assign_map"), 0)
      << "Need graph attribute \"device_assign_map\" in PlaceDevice";
  CHECK_NE(src.attrs.count("device_copy_op"), 0)
      << "Need graph attribute \"device_copy_op\" in PlaceDevice";
  const string& device_group_attr_key = src.GetAttr<string>("device_group_attr_key");
  const Op* copy_op = Op::Get(src.GetAttr<string>("device_copy_op"));
  auto& device_assign_map = src.GetAttr<DeviceAssignMap>("device_assign_map");
  const IndexedGraph& idx = src.indexed_graph();
  DeviceVector device;
  // copy on write semanatics
  if (src.attrs.count("device") != 0) {
    device = src.MoveCopyAttr<DeviceVector>("device");
    CHECK_EQ(device.size(), idx.num_nodes());
  } else {
    device.resize(idx.num_nodes(), -1);
  }

  // Attempt #1: Place node by its given group or its input entry.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    auto it = inode.source->attrs.dict.find(device_group_attr_key);
    if (it != inode.source->attrs.dict.end()) {
      // If the node has group name in the attribute, then place this node
      // on the device associated with that group.
      const string& device_group = it->second;
      auto dit = device_assign_map.find(device_group);
      CHECK(dit != device_assign_map.end())
          << "The device assignment not found for group " << device_group;
      device[nid] = dit->second;
    } else {
      // Search its input entries. Place the node together with any of
      // its input entry.
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        if (device[e.node_id] != -1) {
          device[nid] = device[e.node_id];
          break;
        }
      }
    }
  }
  // Attempt #2: Place node by its output entry.
  for (int nid = idx.num_nodes() - 1; nid >= 0; --nid) {
    const IndexedGraph::Node& inode = idx[nid];
    if (device[nid] == -1) {
      // If the placement of this node is unknown, we could not place its
      // input node according to it.
      continue;
    }
    for (const IndexedGraph::NodeEntry& e : inode.inputs) {
      if (device[e.node_id] == -1) {
        // If any of its input node is not placed, place it together with
        // the currrent node.
        device[e.node_id] = device[nid];
      }
    }
  }
  // Final attempt: Place all other unknown nodes to device 0.
  bool has_multiple_device = false;
  for (uint32_t nodeid = 0; nodeid < idx.num_nodes(); ++nodeid) {
    if (device[nodeid] == -1) {
      device[nodeid] = 0;
    }
    if (device[nodeid] != device[0]) {
      has_multiple_device = true;
    }
  }

  if (!has_multiple_device) {
    src.attrs.erase("device_group_attr_key");
    src.attrs.erase("device_assign_map");
    src.attrs.erase("device_copy_op");
    src.attrs["device"] = std::make_shared<any>(std::move(device));
    return src;
  }

  map<tuple<uint32_t, uint32_t, int>, NodePtr> copy_map;
  // A map from the original node to the node in the new graph.
  vector<NodePtr> new_node_map(idx.num_nodes(), nullptr);
  // A map from the node in the new graph to its assigned device id.
  unordered_map<const Node*, int> new_device_map;
  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");

  // Insert copy node.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const int dev_id = device[nid];
    const auto& inode = idx[nid];

    // Check if any mutable input entry is required to be copy to remote devices.
    if (!inode.source->is_variable() && fmutate_inputs.count(inode.source->op())) {
      for (uint32_t index : fmutate_inputs[inode.source->op()](inode.source->attrs)) {
        const auto& mutate_in_ent = inode.inputs[index];
        CHECK(new_node_map[mutate_in_ent.node_id] == nullptr
            && dev_id == device[mutate_in_ent.node_id])
          << " mutable input entry should not go across device"
          << " op=" << inode.source->op()->name
          << " mutate_input_index=" << index;
      }
    }

    // Whether a new node should be created in the new graph to represent this node,
    // or we could just reuse the original node in the new graph.
    bool require_new_node = false;
    
    // If this node is not a copy node. Loop all the node's input entry.
    // If the input is not placed on the same device with this node, we need to
    // create a new node to represent this node in the new graph.
    if (inode.source->op() != copy_op) {
      for (const auto& e : inode.inputs) {
        if (dev_id != device[e.node_id]) {
          require_new_node = true;
          break;
        }
      }
    }
    // Loop all the node's input entries. If any of its inputs is a new node,
    // also create a new node for this.
    if (!require_new_node) {
      for (const auto& e : inode.inputs) {
        if (new_node_map[e.node_id] != nullptr) {
          require_new_node = true;
          break;
        }
      }
    }
    // Loop all the node's control dependencies. If its dependent node is a new
    // one, also create a new node.
    if (!require_new_node) {
      for (const uint32_t cid : inode.control_deps) {
        if (new_node_map[cid] != nullptr)  {
          require_new_node = true;
          break;
        }
      }
    }
    CHECK(!(inode.source->is_variable() && require_new_node))
      << "Variable node should not be changed during PlaceDevice pass.";

    if (require_new_node) {
      // Create the new node with all its attributes the same as the old one.
      NodePtr new_node = Node::Create();
      new_node->attrs = inode.source->attrs;
      new_node->inputs.reserve(inode.inputs.size());
      for (size_t i = 0; i < inode.inputs.size(); ++i) {
        const IndexedGraph::NodeEntry& in_ent = inode.inputs[i];
        // New input entry.
        const NodeEntry& new_in_ent = new_node_map[in_ent.node_id]?
          NodeEntry{new_node_map[in_ent.node_id], in_ent.index, 0} :
          inode.source->inputs[i];
        if (dev_id != device[in_ent.node_id] && inode.source->op() != copy_op) {
          // Input device and node device is different. Insert copy node.
          auto copy_key = std::make_tuple(in_ent.node_id, in_ent.index, dev_id);
          auto it = copy_map.find(copy_key);
          if (it != copy_map.end()) {
            // The copy node has already been created. This happens when the NodeEntry
            // is used by multiple downstream nodes and they are all on another devices.
            NodePtr copy_node = it->second;
            new_node->inputs.emplace_back(NodeEntry{copy_node, 0, 0});
          } else {
            // Create a new copy node.
            NodePtr copy_node = Node::Create();
            ostringstream os;
            os << inode.source->inputs[i].node->attrs.name << "_"
               << in_ent.index <<"_copy";
            copy_node->attrs.op = copy_op;
            copy_node->attrs.name = os.str();
            // Connect copy node to the input entry.
            copy_node->inputs.push_back(new_in_ent);
            if (copy_node->attrs.op->attr_parser != nullptr) {
              copy_node->attrs.op->attr_parser(&(copy_node->attrs));
            }
            copy_map[copy_key] = copy_node;
            new_device_map[copy_node.get()] = dev_id;
            // Connect the new node to the copy node.
            new_node->inputs.emplace_back(
                NodeEntry{std::move(copy_node), 0, 0});
          }
        } else {
          // Simply connect new node to the input entry.
          new_node->inputs.push_back(new_in_ent);
        }
      }
      new_node->control_deps.reserve(inode.control_deps.size());
      for (size_t i = 0; i < inode.control_deps.size(); ++i) {
        const uint32_t cid = inode.control_deps[i];
        if (new_node_map[cid] != nullptr) {
          new_node->control_deps.push_back(new_node_map[cid]);
        } else {
          new_node->control_deps.push_back(inode.source->control_deps[i]);
        }
      }
      new_device_map[new_node.get()] = dev_id;
      new_node_map[nid] = std::move(new_node);
    } else {
      new_device_map[inode.source] = dev_id;
    }
  }
  // Make the new graph.
  Graph ret;
  for (const NodeEntry& e : src.outputs) {
    if (new_node_map[idx.node_id(e.node.get())] != nullptr) {
      ret.outputs.emplace_back(
          NodeEntry{new_node_map[idx.node_id(e.node.get())], e.index, e.version});
    } else {
      ret.outputs.emplace_back(e);
    }
  }
  const auto& retidx = ret.indexed_graph();
  DeviceVector new_device_vec(retidx.num_nodes());
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    auto source = retidx[nid].source;
    if (new_device_map.count(source) == 0) {
      LOG(FATAL) << "canot find " << source;
    }
    new_device_vec[nid] = new_device_map.at(source);
  }
  if (src.attrs.count("shape") != 0) {
    RemapEntryAttributes<TShape>(src, &ret, new_node_map, "shape");
  }
  if (src.attrs.count("dtype") != 0) {
    RemapEntryAttributes<int>(src, &ret, new_node_map, "dtype");
  }
  //for (size_t i = 0; i < ret.indexed_graph().num_nodes(); ++i) {
    //LOG(INFO) << "Node #" << i << " " << ret.indexed_graph()[i].source->attrs.name << " on " << new_device_vec[i];
  //}

  // Generate dot graph.
  cout << "digraph {" << endl;
  vector<bool> touched_nodes(retidx.num_nodes(), false);
  map<int, vector<uint32_t>> dev2nodes;
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    dev2nodes[new_device_vec[nid]].push_back(nid);
  }
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    const auto& n = retidx[nid];
    for (const auto& in : n.inputs) {
      cout << "\tn" << in.node_id << " -> n" << nid << endl;
      touched_nodes[nid] = true;
      touched_nodes[in.node_id] = true;
    }
  }
  for (const auto& kv : dev2nodes ) {
    const int dev = kv.first;
    const auto& nodes = kv.second;
    cout << "\tsubgraph cluster" << dev << " {" << endl;
    cout << "\t\tlabel=\"device #" << dev << "\"" << endl;
    for (uint32_t nid : nodes) {
      if (!touched_nodes[nid]) {
        continue;
      }
      cout << "\t\tn" << nid << " [label=\""
        << "n" << nid << "_" << retidx[nid].source->attrs.name << "\"]" << endl;
    }
    cout << "\t}" << endl;
  }
  cout << "}" << endl;

  ret.attrs["device"] = std::make_shared<any>(std::move(new_device_vec));
  return ret;
}

NNVM_REGISTER_PASS(PlaceDevice)
.describe("Infer the device type of each operator."\
          "Insert a copy node when there is cross device copy")
.set_body(PlaceDevice)
.set_change_graph(true)
.provide_graph_attr("device")
.depend_graph_attr("device_group_attr_key")
.depend_graph_attr("device_assign_map")
.depend_graph_attr("device_copy_op");

DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);

}  // namespace
}  // namespace pass
}  // namespace nnvm
