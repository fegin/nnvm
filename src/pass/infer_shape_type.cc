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
size_t InferOneNode(Graph& graph,
                    uint32_t nid,
                    FIsNone fis_none,
                    const string& infer_name,
                    const string& shape_attr_key,
                    const AttrType& default_val,
                    vector<AttrType>& inferred) {
  static auto& finfer_shape =
      Op::GetAttr<FInferNodeEntryAttr<AttrType>>(infer_name);
  const IndexedGraph& idx = graph.indexed_graph();
  const auto& inode = idx[nid];
  const uint32_t num_inputs = inode.inputs.size();
  const uint32_t num_outputs = inode.source->num_outputs();
  size_t num_unknown = 0;
  // Temp space for shape inference.
  vector<AttrType> ishape, oshape;
  if (inode.source->is_variable()) {
    // variable node. no operator. only one output entry.
    CHECK(inode.source->op() == nullptr);
    CHECK_EQ(num_outputs, 1);
    const uint32_t out_ent_id = idx.entry_id(nid, 0);
    if (shape_attr_key.length() != 0 && fis_none(inferred[out_ent_id])) {
      auto it = inode.source->attrs.dict.find(shape_attr_key);
      if (it != inode.source->attrs.dict.end()) {
        istringstream is(it->second);
        CHECK(is >> inferred[out_ent_id]) << "invalid attribute";
      }
    }
  } else if (finfer_shape.count(inode.source->op())) {
    // forward operator inference.
    ishape.resize(num_inputs, default_val);
    for (uint32_t i = 0; i < ishape.size(); ++i) {
      ishape[i] = inferred[idx.entry_id(inode.inputs[i])];
    }
    oshape.resize(num_outputs, default_val);
    for (uint32_t i = 0; i < oshape.size(); ++i) {
      oshape[i] = inferred[idx.entry_id(nid, i)];
    }
    
    // call inference function of the operator.
    bool forward_known = finfer_shape[inode.source->op()](
        inode.source->attrs, &ishape, &oshape);
    if (!forward_known) {
      ++num_unknown;
    }

    //for (uint32_t i = 0; i < ishape.size(); ++i) {
    //  if (fis_none(ishape[i])) {
    //    LOG(INFO) << "Node#" << nid << "#i" << i << " Entry#" << idx.entry_id(inode.inputs[i])
    //      << " is none.";
    //  }
    //}
    //for (uint32_t i = 0; i < oshape.size(); ++i) {
    //  if (fis_none(oshape[i])) {
    //    LOG(INFO) << "Node#" << nid << "#o" << i << " Entry#" << idx.entry_id(nid, i)
    //      << " is none.";
    //  }
    //}

    // save to the result map.
    for (uint32_t i = 0; i < num_inputs; ++i) {
      inferred[idx.entry_id(inode.inputs[i])] = ishape[i];
    }
    for (uint32_t i = 0; i < num_outputs; ++i) {
      inferred[idx.entry_id(nid, i)] = oshape[i];
    }
  }
  return num_unknown;
}

template<typename AttrType, typename FIsNone>
Graph InferAttr(Graph&& ret,
                const AttrType default_val,
                const string& infer_name,
                const string& input_name,
                const string& attr_key_name,
                const string& attr_name,
                const string& unknown_name,
                FIsNone fis_none) {
  using AttrVector = vector<AttrType>;
  const IndexedGraph& idx = ret.indexed_graph();
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

  // number of completed nodes
  size_t min_num_unknown = idx.num_nodes();
  bool forward = true;
  while (true) {
    size_t num_unknown = 0;
    if (forward) {
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        num_unknown += InferOneNode(ret, nid, fis_none, infer_name,
            shape_attr_key, default_val, inferred);
      }
    } else {
      for (int nid = idx.num_nodes() - 1; nid >= 0; --nid) {
        num_unknown += InferOneNode(ret, nid, fis_none, infer_name,
            shape_attr_key, default_val, inferred);
      }
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
    LOG(INFO) << "#unknown=" << num_unknown;
    if (num_unknown == min_num_unknown) {
      break;
    }
    min_num_unknown = num_unknown;
    forward = !forward;
  }

  //for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
  //  const auto& inode = idx[nid];
  //  for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
  //    const uint32_t eid = idx.entry_id(nid, i);
  //    LOG(INFO) << "Node#" << nid << "(" << inode.source->attrs.name
  //      << ")#" << i << " entry#" << eid << ": " << inferred[eid];
  //  }
  //}
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(inferred));
  // Number of entries that could not be inferred from this pass.
  LOG(INFO) << "Unknown shape/type: " << min_num_unknown;
  ret.attrs[unknown_name] = std::make_shared<any>(min_num_unknown);
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
