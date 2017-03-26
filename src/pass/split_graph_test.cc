#include <queue>
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

Graph SplitGradientTest(Graph src) {
  const auto& forward_address = src.GetAttr<std::string>("forward_address");
  const auto& backward_address = src.GetAttr<std::string>("backward_address");
  const uint32_t num_forward_outputs = src.GetAttr<uint32_t>("num_forward_outputs");
  AddressVector addresses;
  const auto& idx = src.indexed_graph();
  bool is_backward = true;
  for (uint32_t nid = 0, output_idx = 0; nid < idx.num_nodes(); nid++) {
    if (is_backward) {
      addresses.push_back(backward_address);
    } else {
      addresses.push_back(forward_address);
      if (idx.outputs()[output_idx].node_id == nid) {
        output_idx += 1;
        if (output_idx >= num_forward_outputs) {
          is_backward = true;
        }
      }
    }
  }
  src.attrs["address"] = std::make_shared<dmlc::any>(std::move(addresses));
}

NNVM_REGISTER_PASS(SplitGradientTest)
.describe("Just a test.")
.set_body(SplitGradientTest)
.set_change_graph(false)
.provide_graph_attr("address")
.depend_graph_attr("forward_address")
.depend_graph_attr("backward_address")
.depend_graph_attr("num_forward_outputs");
}  // namespace
}  // namespace pass
}  // namespace nnvm
