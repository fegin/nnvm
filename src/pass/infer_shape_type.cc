/*!
 *  Copyright (c) 2016 by Contributors
 * \file infer_shape.cc
 * \brief Inference the shapes given existin information.
 */
#include <nnvm/pass.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

template<typename AttrType, typename FIsNone>
Graph InferAttr(Graph &&ret,
                const AttrType default_val,
                const string& infer_name,
                const string& input_name,
                const string& attr_key_name,
                const string& attr_name,
                const string& unknown_name,
                FIsNone fis_none) {
  using AttrVector = vector<AttrType>;
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<FInferNodeEntryAttr<AttrType>>(infer_name);
  //static auto& backward_map =
      //Op::GetAttr<FBackwardOutToInIndex>("FBackwardOutToInIndex");
  // reshape shape vector
  AttrVector inferred(idx.num_node_entries(), default_val);

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "More provided shapes than number of arguments.";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      inferred[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
    // erase the provided arguments
    ret.attrs.erase(input_name);
  }
  string shape_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    shape_attr_key = ret.GetAttr<string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }

  // Temp space for shape inference.
  vector<AttrType> ishape, oshape;
  // number of completed nodes
  size_t num_unknown = 0;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (shape_attr_key.length() != 0 && fis_none(inferred[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          istringstream is(it->second);
          CHECK(is >> inferred[out_ent_id]) << "Invalid attribute";
        }
      }
    } else if (finfer_shape.count(inode.source->op())) {
      // Forward operator inference.
      ishape.resize(num_inputs, default_val);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = inferred[idx.entry_id(inode.inputs[i])];
      }
      oshape.resize(num_outputs, default_val);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = inferred[idx.entry_id(nid, i)];
      }
      // Call inference function of the operator.
      bool forward_known = finfer_shape[inode.source->op()](
          inode.source->attrs, &ishape, &oshape);
      if (!forward_known) {
        ++num_unknown;
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        inferred[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        inferred[idx.entry_id(nid, i)] = oshape[i];
      }
    }
    /*else if (backward_map.count(inode.source->op())) {
      // Backward operator inference.
      CHECK_GE(inode.control_deps.size(), 1)
          << "BackwardOp need to have control_deps to its forward op";
      const uint32_t fnode_id = inode.control_deps[0];
      const IndexedGraph::Node& fnode = idx[fnode_id];
      // Inference the outputs of backward operator (equal to the inputs
      // of its corresponding forward operator).
      const vector<uint32_t>& out_map =
          backward_map[inode.source->op()](inode.source->attrs);
      bool known = true;
      for (size_t i = 0; i < out_map.size(); ++i) {
        CHECK_LT(out_map[i], fnode.inputs.size());
        const uint32_t fwd_in_ent_id = idx.entry_id(fnode.inputs[out_map[i]]);
        const uint32_t bwd_out_ent_id = idx.entry_id(nid, i);
        inferred[bwd_out_ent_id] = inferred[fwd_in_ent_id];
        if (fis_none(inferred[bwd_out_ent_id])) {
          // Still unknown due to the forward shape is also unknown.
          known = false;
        }
      }
      num_unknown += !known;
    }*/
  }
  // Inference & check shapes using gradient entry mapping if available.
  if (ret.attrs.count("forward2backward") != 0) {
    const unordered_map<uint32_t, vector<uint32_t>>& forward2backward
      = ret.GetAttr<unordered_map<uint32_t, vector<uint32_t>>>("forward2backward");
    for (const auto& fwd2bwd : forward2backward) {
      const uint32_t fwd_ent_id = fwd2bwd.first;
      for (const uint32_t bwd_ent_id : fwd2bwd.second) {
        if (fis_none(inferred[bwd_ent_id])) {
          inferred[bwd_ent_id] = inferred[fwd_ent_id];
        } else {
          CHECK_EQ(inferred[bwd_ent_id], inferred[fwd_ent_id])
            << inferred[bwd_ent_id] << " v.s. " << inferred[fwd_ent_id]
            << " Backward entry (#" << bwd_ent_id << ") should have the same infer value"
            << " with its corresponding forward (#" << fwd_ent_id << ") entry.";
        }
      }
    }
  }


  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
      const uint32_t eid = idx.entry_id(nid, i);
      LOG(INFO) << inode.source->attrs.name << "#" << i << ": " << inferred[eid];
    }
  }
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(inferred));
  // Number of entries that could not be inferred from this pass.
  LOG(INFO) << "Unknown shape/type: " << num_unknown;
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

NNVM_REGISTER_PASS(InferShape)
.describe("Infer the shape of each node entries.")
.set_body([](Graph ret) {
    return InferAttr<TShape>(
        std::move(ret), TShape(),
        "FInferShape", "shape_inputs", "shape_attr_key",
        "shape", "shape_num_unknown_nodes",
        [](const TShape& s) { return s.ndim() == 0; });
  })
.set_change_graph(false)
.provide_graph_attr("shape");

NNVM_REGISTER_PASS(InferType)
.describe("Infer the dtype of each node entries.")
.set_body([](Graph ret) {
    return InferAttr<int>(
        std::move(ret), 0,
        "FInferType", "dtype_inputs", "dtype_attr_key",
        "dtype", "dtype_num_unknown_nodes",
        [](const int t) { return t == -1; });
  })
.set_change_graph(false)
.provide_graph_attr("dtype");

DMLC_JSON_ENABLE_ANY(ShapeVector, list_shape);
DMLC_JSON_ENABLE_ANY(DTypeVector, list_int);
DMLC_JSON_ENABLE_ANY(size_t, size_t);

}  // namespace
}  // namespace pass
}  // namespace nnvm
