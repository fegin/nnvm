/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */
#ifndef NNVM_PASS_PARTITION_H_
#define NNVM_PASS_PARTITION_H_

#include <nnvm/base.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/scheme.h>
#include <algorithm>
#include <queue>
#include <sstream>

#include "./allreduce.h"
#include "./grid.h"
#include "./tiling.h"

namespace nnvm {
namespace pass {

class GraphPartitioner {
 public:
  GraphPartitioner(const Tiling& tiling, Graph* src,
      const std::string& comm_name, size_t num_devices):
    tiling_(tiling), src_graph_(src), num_devices_(num_devices) {
    comm_planner_ = CommPlanner::CreatePlanner(comm_name);
    copy_op_ = Op::Get("_CrossDeviceCopy");
  }

  void SetOversharding(bool flag) {
    oversharding_ = flag;
  }

  void SetUseFusedConversion(bool flag) {
    use_fused_conversion_ = flag;
  }

  void SetDefaultGraph(const std::string& group) {
    default_group_ = group;
  }

  void SetCopyOp(const Op* copy_op) {
    copy_op_ = copy_op;
  }

  Graph Run();

 private:
  static std::string DevName(size_t device_group_id) {
    return "group:" + std::to_string(device_group_id);
  }

  void AssignDevice(NodePtr node, size_t device_group_id) const;

  void AssignDevice(NodePtr node, const std::string& device_name) const;

  void AssignDefaultGroup(NodePtr node) const;

  std::vector<NodeEntry> SplitEntry(const NodeEntry& from,
                                    const TShape& ret_shape,
                                    const std::string& prefix,
                                    size_t num_args, size_t dim,
                                    const std::string& device_name);

  NodeEntry ConcatEntry(const std::vector<NodeEntry>& from,
                        const TShape& ret_shape,
                        const std::string& prefix, size_t dim,
                        const std::string& device_name);

  void BroadcastEntries(const std::vector<int>& src_dev, const std::vector<int>& tgt_dev,
                        const TShape& shape, std::vector<NodeEntry>* dev_entries);

  void AllReduceBlocks(const std::vector<const Block*>& inputs,
                       const std::vector<Block*>& outputs,
                       const TShape& shape);

  void AllShuffleBlocks(const std::vector<const Block*>& inputs,
                        const std::vector<Block*>& outputs,
                        const TShape& shape);

  void AllReduce(const Grid& input, Grid* output);

  // Create conversion operations between two grids. The conversion happens in three
  // phases: split; allreduce/shuffle; concat.
  void ConvertGrid(const Grid& from, Grid* to);

  // Create conversion operations between two grids. The conversion
  // uses an operator that fuses split, allreduce/shuffle and
  // concat phases of normal ConvertGrid.
  // Note that the fused operator does not need extra buffer for
  // split and concat. It launches kernel that reads data directly
  // from other GPUs.
  // Note: can only be enabled on single-machine-multi-GPUs and when
  //       UVA is supported.
  void FuseConvertGrid(const Grid& from, Grid* to);

  // Connect the input grids and output grids by operators. It follows the idea of
  // recursive-partitionable operator.
  // Note that the input grids could be empty, in which the given operator should
  // also be nullptr. This means the operator is a variable operation.
  void PerformOp(const std::vector<const Grid*>& inputs,
                 const std::vector<Grid*>& outputs,
                 const std::vector<NodePtr>& nodes);

  void SplitVariableGrid(const TShape& shape, const NodeEntry& entry, Grid* to_grid);

  NodeEntry ConcatVariableGrid(const NodeEntry& origin_entry, const Grid& from_grid);

 private:
  const Tiling& tiling_;
  Graph* src_graph_;
  std::unique_ptr<CommPlanner> comm_planner_;
  const size_t num_devices_;

  bool oversharding_{false};
  bool use_fused_conversion_{false};
  std::string default_group_;
  const Op* copy_op_ = nullptr;

  std::unordered_map<NodePtr, std::vector<TShape>> node_output_shapes_;
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_PARTITION_H_
