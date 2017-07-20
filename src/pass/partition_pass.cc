/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition_pass.cc
 * \brief Auto-partition dataflow graph
 */
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>

#include "./partition.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {

// Visitor function for PrintPass.
void PrintPassVisitor(const NodePtr& n) {
  if (n->op()) {
    ostringstream oss;
    for (const auto& map_pair : n->attrs.dict) {
      oss << map_pair.first << " : " << map_pair.second << ", ";
    }
    LOG(INFO) << "PrintPass: Node: \"" << n->attrs.name << "\"; Op \""
              << n->op()->name << "\"; Attrs: {" << oss.str() << "}";
  }
}

int GetNumCuts(int num_devices) {
  CHECK_GT(num_devices, 1) << "Must have more than two devices.";
  int num_cut = 0;
  while(num_devices > 1) {
    CHECK_EQ(num_devices % 2, 0)
      << "Currently only support number of devices equal to 2^x";
    num_devices /= 2;
    ++num_cut;
  }
  return num_cut;
}

}  // namespace

// Pass function that simply print the names of all operators in the graph.
Graph PrintPass(Graph src) {
  DFSVisit(src.outputs, PrintPassVisitor);
  return src;
}

NNVM_REGISTER_PASS(PrintPass)
.describe("Print names of all operators in the graph.")
.set_body(PrintPass)
.set_change_graph(false);


Graph PartitionPass(Graph src) {
  // TODO
  CHECK_NE(src.attrs.count("forward2backward"), 0) 
    << "Gradient entry mapping information is required.";
  CHECK_NE(src.attrs.count("backward2forward"), 0) 
    << "Gradient entry mapping information is required.";
  CHECK_NE(src.attrs.count("num_devices"), 0)
    << "Require number of devices for partitioning";
  const unordered_map<uint32_t, std::vector<uint32_t>>& backward2forward =
    src.GetAttr<unordered_map<uint32_t, std::vector<uint32_t>>>("backward2forward");
  const string& tiling_type = dmlc::GetEnv("TOFU_TILING_TYPE", string("kcuts"));
  const int oversharding = dmlc::GetEnv("TOFU_OVERSHARDING", 0);
  const int num_devices = src.GetAttr<int>("num_devices");
  const int num_cuts = GetNumCuts(num_devices);
  const string& default_group = src.GetAttr<string>("default_group");
    
  LOG(INFO) << "Oversharding status=" << oversharding;
  LOG(INFO) << "Number of cuts: " << num_cuts;

  const IndexedGraph& graph = src.indexed_graph();
  const uint32_t start_node_id = graph.node_id(src.outputs[0].node.get());
  // Construct equal set from gradient information. All output gradient entry should have the
  // same partition scheme with its corresponding input entry.
  vector<pair<uint32_t, uint32_t>> equal;
  for (const NodeEntry& out_ent : src.outputs) {
    const uint32_t out_ent_id = graph.entry_id(out_ent);
    if (backward2forward.find(out_ent_id) != backward2forward.end()) {
      // This is a gradient output entry. Add equilibrium of it and its forward entry.
      const uint32_t fwd_ent_id = backward2forward.at(out_ent_id)[0];
      equal.emplace_back(out_ent_id, fwd_ent_id);
    }
  }
  // Set all element-wise operators' input and output entries to have the same partition.
  /*for (uint32_t nid = 0; nid < graph.num_nodes(); ++nid) {
    const Node* node = graph[nid].source;
    if (node->is_variable()) {
      continue;
    }
    const auto& opname = node->op()->name;
    if (opname == "Activation" ||
        opname == "elemwise_add" ||
        opname == "_backward_Activation" ||
        opname == "ElementWiseSum" ||
        opname == "_backward_add") {
      CHECK_GT(node->inputs.size(), 0);
      const uint32_t inent0_id = graph.entry_id(node->inputs[0]);
      for (size_t i = 1; i < node->inputs.size(); ++i) {
        equal.emplace_back(graph.entry_id(node->inputs[i]), inent0_id);
      }
      for (size_t i = 0; i < node->num_outputs(); ++i) {
        equal.emplace_back(graph.entry_id(nid, i), inent0_id);
      }
    }
  }*/

  NodeEntryGroups groups(graph.num_node_entries(), equal);

  // Call BFS.
  BFS lvls(&src, &groups);
  lvls.Run(start_node_id);
  lvls.Print();
  // TODO(minjie): chaos ownership
  //NeuralLevels lvls(&src, &groups);
  //lvls.Run();
  //lvls.Print();

  Tiling* tiling = nullptr;
  if (tiling_type == "kcuts") {
    // Cut algorithm.
    CutAlgorithm* algo = new CutAlgorithm(&src, lvls, groups);
    cost_t total_cost = 0;
    total_cost = algo->KCuts(num_cuts);
    algo->Print();
    LOG(INFO) << "Total K-cuts cost: " << total_cost;
    if (oversharding) {
      LOG(INFO) << "Oversharding enabled";
      Tiling* overshard_tiling = new DataParallelism(&src, groups, 2);
      tiling = new MergeTiling(&src, algo, overshard_tiling);
    } else {
      tiling = algo;
    }
  } else if (tiling_type == "datapar") {
    // Data parallelism
    tiling = new DataParallelism(&src, groups, num_devices);
  } else if (tiling_type == "modelpar") {
    // Model parallelism
    tiling = new ModelParallelism(&src, groups, num_devices);
  }

  // Graph partitioner.
  CHECK_NOTNULL(tiling);
  const int num_partitions = oversharding? num_devices * 2 : num_devices;
  GraphPartitioner pttn(
      *tiling, &src,
      CommPlanner::kDefaultPlanner,
      num_partitions);
  pttn.SetOversharding(oversharding);
  pttn.SetDefaultGraph(default_group);

  const Graph& ret = pttn.Run();

  delete tiling;
  return ret;
}

NNVM_REGISTER_PASS(PartitionPass)
.describe("Partition tensors in graph and place them to multiple devices.")
.set_body(PartitionPass)
.depend_graph_attr("shape")  // Shape information from InferShapePass.
.depend_graph_attr("forward2backward")  // Gradient information from GradientPass.
.depend_graph_attr("backward2forward")  // Gradient information from GradientPass.
.depend_graph_attr("num_devices")  // Number of devices
.depend_graph_attr("default_group")
.depend_op_attr("FAlignedSchemes")  // Require op to provide aligned schemes.
.set_change_graph(true);


}  // namespace pass
}  // namespace nnvm
