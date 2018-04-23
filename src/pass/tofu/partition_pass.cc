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
#include "./tiling.h"
#include "./search_graph.h"

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
  const int fused_conversion = dmlc::GetEnv("TOFU_FUSED_CONVERSION", 0);
  const int use_bfs = dmlc::GetEnv("TOFU_USE_BFS_LEVEL", 0);

  const int num_devices = src.GetAttr<int>("num_devices");
  const int num_cuts = GetNumCuts(num_devices);
  const string& default_group = src.GetAttr<string>("default_group");
  const string& copy_op_name = src.GetAttr<string>("copy_op_name");
    
  LOG(INFO) << "Tiling type: " << tiling_type;
  LOG(INFO) << "Oversharding status=" << oversharding;
  LOG(INFO) << "Fused conversion status=" << fused_conversion;
  LOG(INFO) << "Number of cuts: " << num_cuts;

  const IndexedGraph& graph = src.indexed_graph();
  // Construct equal set from gradient information. All output gradient entry should have the
  // same partition scheme with its corresponding input entry.
  //vector<pair<uint32_t, uint32_t>> equal;
  //for (const NodeEntry& out_ent : src.outputs) {
  //  const uint32_t out_ent_id = graph.entry_id(out_ent);
  //  if (backward2forward.find(out_ent_id) != backward2forward.end()) {
  //    // This is a gradient output entry. Add equilibrium of it and its forward entry.
  //    const uint32_t fwd_ent_id = backward2forward.at(out_ent_id)[0];
  //    equal.emplace_back(out_ent_id, fwd_ent_id);
  //  }
  //}

  MegaGraph mg(&src);
  //mg.Print();
  mg.MergeElementwise();
  mg.MergeWeightAndGradients();
  mg.MergeRNNSteps();
  const auto& equal = mg.entry_equals();

  NodeEntryGroups groups(graph.num_node_entries(), equal);

  // TODO(minjie): chaos ownership
  Levels* lvls = nullptr;
  if (use_bfs) {
    const uint32_t start_node_id =
      graph.node_id(src.outputs[0].node.get());
    BFS* bfslvls = new BFS(&src, &groups);
    bfslvls->Run(start_node_id);
    bfslvls->Print();
    lvls = bfslvls;
  } else {
    NeuralLevels* nnlvls = new NeuralLevels(&src, &groups);
    nnlvls->Run();
    nnlvls->Print();
    lvls = nnlvls;
  }
  LOG(FATAL) << "!!!!!";

  unique_ptr<Tiling> tiling = Tiling::Create(
      tiling_type,
      &src,
      *lvls,
      groups,
      num_devices);
  tiling->Run();
  tiling->Print();
  if (oversharding) {
    LOG(INFO) << "Oversharding enabled";
    //Tiling* overshard_tiling = new DataParallelism(&src, groups, 2);
    //tiling = new MergeTiling(&src, algo, overshard_tiling);
    LOG(FATAL) << "Disable for now";
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
  pttn.SetUseFusedConversion(fused_conversion);
  pttn.SetCopyOp(Op::Get(copy_op_name));

  const Graph& ret = pttn.Run();

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
.depend_graph_attr("user_tiling_json")
.depend_op_attr("FAlignedSchemes")  // Require op to provide aligned schemes.
.set_change_graph(true);

}  // namespace pass
}  // namespace nnvm
