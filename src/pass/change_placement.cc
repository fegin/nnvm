/*!
 *  Copyright (c) 2016 by Contributors
 * \file place_device.cc
 * \brief Inference the device of each operator given known information.
 *  Insert a copy node automatically when there is a cross device.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

// simply logic to place device according to device_group hint
// insert copy node when there is
Graph ChangePlacement(Graph src) {
  CHECK(src.attrs.count("ngpus"))
      << "Need graph attribute \"ngpus\" in PlaceDevice";
  CHECK(src.attrs.count("device_group_attr_key"))
      << "Need graph attribute \"device_group_attr_key\" in PlaceDevice";
  CHECK(src.attrs.count("placement"))
      << "Need graph attribute \"placement\" in PlaceDevice";

  uint32_t ngpus = src.GetAttr<uint32_t>("ngpus");
  std::string device_group_attr_key = src.GetAttr<std::string>("device_group_attr_key");
  auto& placement = src.GetAttr<PlacementVector>("placement");
  std::vector<std::string> device_vector;
  const IndexedGraph& idx = src.indexed_graph();
  std::vector<NodePtr> new_node_map(idx.num_nodes(), nullptr);

  CHECK(placement.size() == idx.num_nodes());
  for (uint32_t i = 0; i < ngpus; i++) {
    std::stringstream sstring;
    sstring << "GPU" << i;
    device_vector.push_back(sstring.str());
  }
  // Change device_group_attr_key to the assigned GPU.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    NodePtr new_node = Node::Create();
    new_node->inputs.reserve(inode.inputs.size());
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      const IndexedGraph::NodeEntry& e = inode.inputs[i];
      CHECK(new_node_map[e.node_id] != nullptr);
      new_node->inputs.emplace_back(
          NodeEntry{new_node_map[e.node_id], e.index, 0});
    }
    for (size_t i = 0; i < inode.control_deps.size(); ++i) {
      uint32_t cid = inode.control_deps[i];
      CHECK(new_node_map[cid] != nullptr);
      new_node->control_deps.push_back(new_node_map[cid]);
    }
    new_node->control_deps.reserve(inode.control_deps.size());
    new_node->attrs = inode.source->attrs;
    new_node->attrs.dict[device_group_attr_key] = device_vector[placement[nid]];
    new_node_map[nid] = std::move(new_node);
  }

  // make the new graph
  Graph ret;
  for (const NodeEntry& e : src.outputs) {
    if (new_node_map[idx.node_id(e.node.get())] != nullptr) {
      ret.outputs.emplace_back(
          NodeEntry{new_node_map[idx.node_id(e.node.get())], e.index, e.version});
    } else {
      ret.outputs.emplace_back(e);
    }
  }
  DeviceAssignMap device_map;
  for (uint32_t i = 0; i < ngpus; i++) {
    device_map[device_vector[i]] = i;
  }
  ret.attrs["device_map"] = std::make_shared<any>(std::move(device_map));
  return ret;
}

NNVM_REGISTER_PASS(ChangePlacement)
.describe("Infer the device type of each operator."\
          "Insert a copy node when there is cross device copy")
.set_body(ChangePlacement)
.set_change_graph(true)
.provide_graph_attr("device_map")
.depend_graph_attr("ngpus")
.depend_graph_attr("device_group_attr_key")
.depend_graph_attr("placement");

DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);

}  // namespace
}  // namespace pass
}  // namespace nnvm
