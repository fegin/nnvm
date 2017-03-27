#include <queue>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

Graph SplitGradientTest(Graph src) {
  const auto& device_group_attr_key =
      src.GetAttr<std::string>("device_group_attr_key");
  const uint32_t num_forward_outputs = src.GetAttr<uint32_t>("num_forward_outputs");
  const auto& idx = src.indexed_graph();
  bool is_backward = false;
  for (uint32_t nid = 0, output_idx = 0; nid < idx.num_nodes(); nid++) {
    const std::unordered_map<std::string, std::string>& dict
        = idx[nid].source->attrs.dict;
    if (is_backward) {
      const_cast<std::unordered_map<std::string, std::string>&>
          (dict)[device_group_attr_key] = "backward";
    } else {
      const_cast<std::unordered_map<std::string, std::string>&>
          (dict)[device_group_attr_key] = "forward";
      if (idx.outputs()[output_idx].node_id == nid) {
        output_idx += 1;
        if (output_idx >= num_forward_outputs) {
          is_backward = true;
        }
      }
    }
  }
  return src;
}

NNVM_REGISTER_PASS(SplitGradientTest)
.describe("Just a test.")
.set_body(SplitGradientTest)
.set_change_graph(false)
.depend_graph_attr("device_group_attr_key")
.depend_graph_attr("num_forward_outputs");
}  // namespace
}  // namespace pass
}  // namespace nnvm
