/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */

#include "./partition.h"

#include <queue>
#include <dmlc/json.h>
#include <nnvm/symbolic.h>

#include "./utils.h"
#include "./geometry.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {
inline void FinalizeNodeCreation(NodePtr node) {
  static int count = 0;
  //cout << "Create! #" << count << ": " << node.get() << " " << node->attrs.name << endl;
  ++count;
  // Parse attributes.
  if (node->attrs.op && node->attrs.op->attr_parser) {
    node->attrs.op->attr_parser(&(node->attrs));
  }
}

#define CHECK_ONDEVICE(ent, dev) \
  CHECK_EQ((ent).node->attrs.dict["__ctx_group__"], "group:" + std::to_string((dev))) \
  << (ent).node->attrs.dict["__ctx_group__:"] << " v.s. " << (dev)

template<typename T>
inline vector<int> GetDevId(const vector<T>& inputs) {
  vector<int> ret(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    ret[i] = inputs[i]->device_group_id;
  }
  return ret;
}
}  // namespace

void GraphPartitioner::AssignDevice(NodePtr node, size_t device_group_id) const {
  node->attrs.dict["__ctx_group__"] = DevName(device_group_id);
}

void GraphPartitioner::AssignDevice(NodePtr node, const std::string& device_name) const {
  node->attrs.dict["__ctx_group__"] = device_name;
}

void GraphPartitioner::AssignDefaultGroup(NodePtr node) const {
  node->attrs.dict["__ctx_group__"] = default_group_;
}

vector<NodeEntry> GraphPartitioner::SplitEntry(
    const NodeEntry& from,
    const TShape& ret_shape,
    const string& prefix,
    size_t num_args, size_t dim,
    const string& device_name) {
  CHECK_GT(num_args, 0);
  CHECK_LT(dim, ret_shape.ndim());
  if (num_args == 1) {
    // Split operation is not needed.
    return {from};
  }
  // Split op name.
  ostringstream oss;
  oss << "_TOFU[" << prefix << "]SPLITd" << dim;
  // Split op.
  const Op* split_op = Op::Get("_backward_Concat");
  NodePtr node = Node::Create();
  node->inputs.push_back(from);
  node->attrs.op = split_op;
  node->attrs.name = oss.str();
  node->attrs.dict["num_args"] = std::to_string(num_args);
  node->attrs.dict["dim"] = std::to_string(dim);
  AssignDevice(node, device_name);
  FinalizeNodeCreation(node);
  // Create output entries.
  vector<NodeEntry> ret;
  CHECK(node_output_shapes_[node].empty());
  for (uint32_t i = 0; i < num_args; ++i) {
    ret.push_back(NodeEntry{node, i, 0});
    // Output shape.
    node_output_shapes_[node].push_back(ret_shape);
  }
  return ret;
}

NodeEntry GraphPartitioner::ConcatEntry(
    const vector<NodeEntry>& from,
    const TShape& ret_shape,
    const string& prefix, size_t dim,
    const string& device_name) {
  CHECK(!from.empty());
  CHECK_LT(dim, ret_shape.ndim());
  if (from.size() == 1) {
    // Concat operation is not needed.
    return from[0];
  }
  const Op* concat_op = Op::Get("Concat");
  // Concat op name.
  ostringstream oss;
  oss << "_TOFU[" << prefix << "]CONCATd" << dim;
  // Concat op.
  NodePtr node = Node::Create();
  node->inputs = from;
  node->attrs.op = concat_op;
  node->attrs.name = oss.str();
  node->attrs.dict["num_args"] = std::to_string(from.size());
  node->attrs.dict["dim"] = std::to_string(dim);
  AssignDevice(node, device_name);
  FinalizeNodeCreation(node);
  // Create output entries.
  NodeEntry to{node, 0, 0};
  CHECK(node_output_shapes_[node].empty());
  node_output_shapes_[node].push_back(ret_shape);
  return to;
}

void GraphPartitioner::BroadcastEntries(
    const vector<int>& src_dev, const vector<int>& tgt_dev,
    const TShape& shape, vector<NodeEntry>* dev_entries) {
  CHECK_EQ(dev_entries->size(), num_devices_);
  const Op* copy_op = Op::Get("_CrossDeviceCopy");
  vector<bool> visited(num_devices_, false);
  for (const int src : src_dev) {
    visited[src] = true;
  }
  const auto& stages = comm_planner_->BroadcastPlan(src_dev, tgt_dev);
  for (size_t stageid = 0; stageid < stages.size(); ++stageid) {
    for (const CommPlanner::Broadcast& bcast : stages[stageid].broadcasts) {
      CHECK(visited[bcast.from]);
      for (const int to : bcast.to) {
        if (to == bcast.from) {
          (*dev_entries)[to] = (*dev_entries)[bcast.from];
        } else {
          CHECK(!visited[to]);
          if (std::find(tgt_dev.begin(), tgt_dev.end(), to) != tgt_dev.end()) {
            // The broadcast target is contained in the output targets.
            (*dev_entries)[to] = (*dev_entries)[bcast.from];
          } else {
            CHECK(false) << "Multi-stage broadcasting is not allowed right now.";
            NodePtr copy_node = Node::Create();
            copy_node->attrs.op = copy_op;
            copy_node->attrs.name = "_TOFU[red]BCAST" + std::to_string(stageid);
            copy_node->inputs.push_back((*dev_entries)[bcast.from]);
            AssignDevice(copy_node, to);
            FinalizeNodeCreation(copy_node);
            // Shape.
            CHECK(node_output_shapes_[copy_node].empty());
            node_output_shapes_[copy_node].push_back(shape);
            // Update the node entry of the target node.
            (*dev_entries)[to] = NodeEntry{copy_node, 0, 0};
          }
          visited[to] = true;
        }
      }
    }
  }
  for (const int tgt : tgt_dev) {
    CHECK(visited[tgt]);
    // TODO(minjie): cannot use following sanity check since in reduce, if there is only
    // one in and one output, the entry is directly assigned, leading to entry with different
    // device. Though this is still good since PlaceDevice pass will fix it. It is still
    // better to remove that case and put an explicit copy in it.
    //CHECK_ONDEVICE((*dev_entries)[tgt], tgt);
  }
}

void GraphPartitioner::AllReduceBlocks(
    const vector<const Block*>& inputs, const vector<Block*>& outputs,
    const TShape& shape) {
  CHECK_GT(inputs.size(), 1);
  const Op* sum_op = Op::Get("ElementWiseSum");

  // Split for balanced allreduce.
  vector<vector<NodeEntry>> splitted(inputs.size());
  // TODO(minjie): The split here should be a FlattenAndSplit because we
  // in fact don't care about the shape but only the length of the array.
  CHECK_EQ(shape[0] % outputs.size(), 0) << shape[0] << " " << outputs.size();
  TShape split_shape = shape;
  split_shape[0] /= outputs.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    splitted[i] = SplitEntry(inputs[i]->entry,
                             split_shape,
                             "red",
                             outputs.size(),
                             0 /*split dim */,
                             DevName(inputs[i]->device_group_id) /*device*/);
  }

  // Multi-stage Allreduce.
  //  - Reduce Phase:
  vector<NodeEntry> final_sum(outputs.size());
  const vector<int>& src_dev = GetDevId(inputs);
  for (size_t i = 0; i < outputs.size(); ++i) {
    vector<NodeEntry> tmp_sum(num_devices_);
    // Initial reduced entries are from the splitted inputs.
    for (size_t j = 0; j < inputs.size(); ++j) {
      tmp_sum[inputs[j]->device_group_id] = splitted[j][i];
    }
    // Get reduce plan.
    const uint32_t tgt_dev = outputs[i]->device_group_id;
    const vector<CommPlanner::ReduceStage>& stages = comm_planner_->ReducePlan(
        src_dev, tgt_dev);
    // Perform multi-stage reduce.
    for (size_t stageid = 0; stageid < stages.size(); ++stageid) {
      if (stageid == stages.size() - 1) {
        // Final stage must sum to the target device.
        CHECK(stages[stageid].reduces.size() == 1 &&
              stages[stageid].reduces[0].to == tgt_dev);
      }
      for (const CommPlanner::Reduce& red : stages[stageid].reduces) {
        if (red.from.size() == 1) {
          // Only one input, just directly use that entry without summation.
          tmp_sum[red.to] = tmp_sum[red.from[0]];
        } else {
          // Create sum node.
          NodePtr sum_node = Node::Create();
          // Create input entries.
          for (const size_t procid : red.from) {
            sum_node->inputs.push_back(tmp_sum[procid]);
          }
          sum_node->attrs.op = sum_op;
          sum_node->attrs.name = "_TOFU[red]SUM" + std::to_string(stageid);
          sum_node->attrs.dict["num_args"] = std::to_string(red.from.size());
          AssignDevice(sum_node, red.to);
          FinalizeNodeCreation(sum_node);
          // Shape.
          CHECK(node_output_shapes_[sum_node].empty());
          node_output_shapes_[sum_node].push_back(split_shape);
          // Update the node entry of the target node.
          tmp_sum[red.to] = NodeEntry{sum_node, 0, 0};
        }
      }
    }
    // Save the final sum.
    final_sum[i] = tmp_sum[tgt_dev];
  }
  // - Broadcast Phase:
  vector<vector<NodeEntry>> to_concat(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    to_concat[i].resize(final_sum.size());
  }
  const vector<int>& tgt_dev = GetDevId(outputs);
  for (size_t i = 0; i < final_sum.size(); ++i) {
    const int src_dev = outputs[i]->device_group_id;
    vector<NodeEntry> tmp_bcast(num_devices_);
    tmp_bcast[src_dev] = final_sum[i];
    BroadcastEntries({src_dev}, tgt_dev, split_shape, &tmp_bcast);
    // Record the broadcast outputs.
    for (size_t j = 0; j < outputs.size(); ++j) {
      to_concat[j][i] = tmp_bcast[tgt_dev[j]];
    }
  }
  
  // Concat.
  for (size_t i = 0; i < outputs.size(); ++i) {
    //for (const auto& concat_ent : to_concat[i]) {
      //CHECK_EQ(concat_ent.node->attrs.dict["ctx_group"],
               //to_concat[i][0].node->attrs.dict["ctx_group"])
        //<< concat_ent.node->attrs.dict["ctx_group"]
        //<< " v.s. " << to_concat[i][0].node->attrs.dict["ctx_group"];
    //}
    outputs[i]->entry = ConcatEntry(to_concat[i],
                                    shape,
                                    "red",
                                    0 /*concat dim*/,
                                    DevName(outputs[i]->device_group_id) /*device id*/);
  }
}

void GraphPartitioner::AllShuffleBlocks(
    const vector<const Block*>& inputs, const vector<Block*>& outputs,
    const TShape& shape) {
  CHECK(!inputs.empty() && !outputs.empty());
  const vector<int>& src_dev = GetDevId(inputs);
  const vector<int>& tgt_dev = GetDevId(outputs);
  vector<NodeEntry> tmp_bcast(num_devices_);
  for (const Block* inblk : inputs) {
    tmp_bcast[inblk->device_group_id] = inblk->entry;
  }
  BroadcastEntries(src_dev, tgt_dev, shape, &tmp_bcast);
  // Copy the broadcast result to the outputs.
  for (Block* outblk : outputs) {
    outblk->entry = tmp_bcast[outblk->device_group_id];
  }
}

void GraphPartitioner::AllReduce(const Grid& input, Grid* output) {
  CHECK_GT(input.TotalNumBlocks(), 0);
  CHECK_EQ(input.num_blocks(), output->num_blocks());
  CHECK_EQ(input.block_shape(), output->block_shape());
  ConstGridIndexMap ingrid_idx(input);
  GridIndexMap outgrid_idx(*output);
  IndexIter iter(input.num_blocks());
  vector<const Block*> input_blocks(input.num_replicates());
  vector<Block*> output_blocks(output->num_replicates());
  // Do allreduce/shuffle for blocks of the same grid index.
  do {
    const TShape& curidx = iter.Get();
    for (size_t repid = 0; repid < input.num_replicates(); ++repid) {
      input_blocks[repid] = &(ingrid_idx.GetBlock(curidx, repid));
    }
    for (size_t repid = 0; repid < output->num_replicates(); ++repid) {
      output_blocks[repid] = &(outgrid_idx.GetBlock(curidx, repid));
    }
    if (input.replicate_is_reduction()) {
      AllReduceBlocks(input_blocks, output_blocks, input.block_shape());
    } else {
      AllShuffleBlocks(input_blocks, output_blocks, input.block_shape());
    }
  } while(iter.Next());
}

void GraphPartitioner::ConvertGrid(const Grid& from, Grid* to) {
  CHECK_EQ(from.shape(), to->shape());
  CHECK(!to->replicate_is_reduction());
  if (from.num_blocks() == to->num_blocks() &&
      from.num_replicates() == to->num_replicates() &&
      !from.replicate_is_reduction()) {
    to->CopyFrom(from);
    return;
  }
  //LOG(INFO) << "Convert from: " << from.num_blocks() << "x" << from.num_replicates()
    //<< " to " << to->num_blocks() << "x" << to->num_replicates();

  // Three phase conversion: split + allreduce/shuffle + concat
  // Note that the split is implemented by _backward_Concat.
  const TShape& max_num_blocks = max(from.num_blocks(), to->num_blocks());

  // Phase: Split
  const TShape& extra_from_cuts = max_num_blocks / from.num_blocks();
  Grid from_split = from;
  for (size_t i = 0; i < extra_from_cuts.ndim(); ++i) {
    if (extra_from_cuts[i] <= 1) {
      continue;
    }
    from_split.InnerSplit(
        Scheme::Cut(i), extra_from_cuts[i],
        [&] (const Block& from, const TShape& from_shape,
             const vector<Block*>& to, const TShape& to_shape) {
          // Split function here.
          const vector<NodeEntry>& splitted =
            SplitEntry(from.entry, to_shape, "convert",
                       to.size(), i /* dim */,
                       DevName(from.device_group_id));
          for (uint32_t idx = 0; idx < to.size(); ++idx) {
            CHECK_EQ(from.device_group_id, to[idx]->device_group_id);
            to[idx]->entry = splitted[idx];
          }
        });
  }
  //cout << "Need to split: " << extra_from_cuts << endl;
  //from.PrettyPrint();
  //cout << "After inner split: " << endl;
  //from_split.PrettyPrint();
  const TShape& extra_to_cuts = max_num_blocks / to->num_blocks();
  for (size_t i = 0; i < extra_to_cuts.ndim(); ++i) {
    if (extra_to_cuts[i] <= 1) {
      continue;
    }
    to->InnerSplit(Scheme::Cut(i), extra_to_cuts[i]);
  }
  CHECK_EQ(from_split.num_blocks(), to->num_blocks());

  // Phase: Allreduce/shuffle
  AllReduce(from_split, to);
  
  // Phase: Concat
  for (int i = extra_to_cuts.ndim() - 1; i >= 0; --i) {
    if (extra_to_cuts[i] > 1) {
      const auto& poped = to->InnerConcat(
          [&] (const vector<const Block*>& from, const TShape& from_shape,
               Block* to, const TShape& to_shape) {
            // Concat function.
            vector<NodeEntry> ent;
            for (auto blk : from) {
              CHECK_EQ(blk->device_group_id, to->device_group_id);
              ent.push_back(blk->entry);
            }
            to->entry = ConcatEntry(ent, to_shape,
              "convert", i /* dim */, DevName(to->device_group_id));
          });
      CHECK(poped.first.type == Scheme::kCut);
      CHECK(poped.first.dim == i);
      CHECK(poped.second == extra_to_cuts[i]);
    }
  }

  if (use_fused_conversion_) {
    FuseConvertGrid(from, to);
  }
}

void GraphPartitioner::FuseConvertGrid(const Grid& from, Grid* to) {
  const Op* split_op = Op::Get("_backward_Concat");
  std::unordered_set<const Node*> root;
  for (size_t i = 0; i < from.TotalNumBlocks(); ++i) {
    root.insert(from.BlockAt(i).entry.node.get());
  }
  for (size_t i = 0; i < to->TotalNumBlocks(); ++i) {
    auto& toblk = to->BlockAt(i);
    std::vector<NodeEntry> blkroot;
    std::unordered_map<const Node*, std::unordered_map<const Node*, Region>> node2region;
    DFSVisitWithRoot({toblk.entry.node.get()}, root,
        [&] (const Node* node, const unordered_set<uint32_t>& oidx) {
          std::unordered_map<const Node*, Region> inreg;
          for (const auto& inent : node->inputs) {
            const Node* innode = inent.node.get();
            if (root.count(innode)) {
              blkroot.push_back(inent);
              inreg[innode] = Region(from.block_shape());
            } else if (innode->op() == split_op) {
              int num_args = std::stoi(innode->attrs.dict.at("num_args"));
              int dim = std::stoi(innode->attrs.dict.at("dim"));
              CHECK(node2region.at(innode).size() == 1);
              for (const auto& kv : node2region.at(innode)) {
                inreg[kv.first] = kv.second.Split(Scheme::Cut(dim), num_args).at(inent.index);
              }
            } else {
              for (const auto& kv : node2region.at(innode)) {
                inreg[kv.first] = kv.second;
              }
            }
          }
          node2region[node] = std::move(inreg);
        });
    //LOG(INFO) << "#Root blks: " << blkroot.size();
    //LOG(INFO) << "Fused: [";
    //LOG(INFO) << "]";
    const auto& dep_regions = node2region.at(toblk.entry.node.get());
    std::vector<TShape> offsets, sizes;
    for (const auto& br : blkroot) {
      offsets.push_back(dep_regions.at(br.node.get()).offset());
      sizes.push_back(dep_regions.at(br.node.get()).shape());
    }
    // Fused op name.
    ostringstream oss;
    oss << "_TOFU_CONVERT";
    NodePtr node = Node::Create();
    // all input entries
    node->inputs = std::move(blkroot);
    node->attrs.op = tofu_fused_convert_op_;
    node->attrs.name = oss.str();
    node->attrs.parsed = std::make_pair(offsets, sizes);
    //node->attrs.dict["num_args"] = std::to_string(node->inputs.size());
    AssignDevice(node, DevName(toblk.device_group_id));
    FinalizeNodeCreation(node);
    toblk.entry = NodeEntry{node, 0, 0};
    CHECK(node_output_shapes_[node].empty());
    node_output_shapes_[node].push_back(to->block_shape());
  }
}

void GraphPartitioner::PerformOp(const vector<const Grid*>& inputs,
                                 const vector<Grid*>& outputs,
                                 const vector<NodePtr>& nodes) {
  // Sub-operators can simply fetch blocks under the same devices
  // since all required data should be fetched to locally already.
  const uint32_t num_devices = nodes.size();
  for (uint32_t dev = 0; dev < num_devices; ++dev) {
    nodes[dev]->inputs.resize(inputs.size());
    for (uint32_t i = 0; i < inputs.size(); ++i) {
      CHECK_EQ(inputs[i]->TotalNumBlocks(), num_devices);
      nodes[dev]->inputs[i] = inputs[i]->BlockAt(dev).entry;
    }
    for (uint32_t i = 0; i < outputs.size(); ++i) {
      CHECK_EQ(outputs[i]->TotalNumBlocks(), num_devices);
      outputs[i]->BlockAt(dev).entry = NodeEntry{nodes[dev], i, 0};
    }
  }
}
  
void GraphPartitioner::SplitVariableGrid(
    const TShape& shape,
    const NodeEntry& entry,
    Grid* to_grid) {
  const int fake_var_split_concat = dmlc::GetEnv("TOFU_FAKE_VAR_SPLIT_CONCAT", 0);
  if (fake_var_split_concat) {
    // Use nop operators and control dependency to simulate the split
    // while no real split/concat/copy happens.
    for (size_t i = 0; i < to_grid->TotalNumBlocks(); ++i) {
      NodePtr zeronode = Node::Create();
      // TODO(minjie): should be zero node.
      zeronode->attrs.op = Op::Get("_TofuFakeVar");
      zeronode->attrs.name = entry.node->attrs.name + "_" + std::to_string(i);
      // Control dependency.
      zeronode->control_deps.push_back(entry.node);
      AssignDevice(zeronode, to_grid->BlockAt(i).device_group_id);
      FinalizeNodeCreation(zeronode);
      // Output entry and shape.
      to_grid->BlockAt(i).entry = {zeronode, 0, 0};
      CHECK(node_output_shapes_[zeronode].empty());
      node_output_shapes_[zeronode].push_back(to_grid->block_shape());
    }
  } else {
    Grid from_grid(shape, entry);
    const TShape& ncuts_per_dim = to_grid->num_blocks();
    const uint32_t nreps = to_grid->num_replicates();
    // Cuts.
    for (size_t i = 0; i < ncuts_per_dim.ndim(); ++i) {
      if (ncuts_per_dim[i] <= 1) {
        continue;
      }
      from_grid.OuterSplit(Scheme::Cut(i), ncuts_per_dim[i],
          [&] (const Block& from, const TShape& from_shape,
               const vector<Block*>& to, const TShape& to_shape) {
            const vector<NodeEntry>& splitted =
              SplitEntry(from.entry,
                         to_shape, "var",
                         to.size(), i /*dim*/,
                         default_group_ /* device */);
            for (uint32_t idx = 0; idx < to.size(); ++idx) {
              to[idx]->entry = splitted[idx];
            }
          });
    }
    // Replicates.
    from_grid.OuterSplit(Scheme::Rep(), nreps);
    for (size_t i = 0; i < from_grid.TotalNumBlocks(); ++i) {
      from_grid.BlockAt(i).device_group_id = i;
    }
    // Copy block entries to the target grid.
    to_grid->CopyFrom(from_grid);
  }
}

NodeEntry GraphPartitioner::ConcatVariableGrid(
    const NodeEntry& original_entry,
    const Grid& from_grid) {
  const int fake_var_split_concat = dmlc::GetEnv("TOFU_FAKE_VAR_SPLIT_CONCAT", 0);
  if (fake_var_split_concat) {
    NodePtr fake_out_node = Node::Create();
    // TODO(minjie): should be zero node.
    fake_out_node->attrs.op = Op::Get("_NoGradient");
    fake_out_node->attrs.name = original_entry.node->attrs.name;
    FinalizeNodeCreation(fake_out_node);
    CHECK(node_output_shapes_[fake_out_node].empty());
    node_output_shapes_[fake_out_node].push_back(from_grid.shape());
    for (size_t i = 0; i < from_grid.TotalNumBlocks(); ++i) {
      // Add control dependencies.
      fake_out_node->control_deps.push_back(from_grid.BlockAt(i).entry.node);
    }
    AssignDefaultGroup(fake_out_node);  // The fake node is assigned to the default group.
    return NodeEntry{fake_out_node, 0, 0};
  } else {
    const TShape& ncuts_per_dim = from_grid.num_blocks();
    const uint32_t nreps = from_grid.num_replicates();
    // First do "fake" split to prepare for concatenation.
    Grid to_grid(from_grid.shape(), vector<Scheme>());
    for (size_t i = 0; i < ncuts_per_dim.ndim(); ++i) {
      if (ncuts_per_dim[i] == 1) {
        continue;
      }
      to_grid.OuterSplit(Scheme::Cut(i), ncuts_per_dim[i]);
    }
    if (nreps > 1) {
      to_grid.OuterSplit(Scheme::Rep(), nreps);
    }

    to_grid.CopyFrom(from_grid);

    // "Concat" replication.
    if (nreps > 1) {
      to_grid.OuterConcat();
    }
    // Concat.
    for (int i = ncuts_per_dim.ndim() - 1; i >= 0; --i) {
      if (ncuts_per_dim[i] <= 1) {
        continue;
      }
      const auto& poped = to_grid.OuterConcat(
        [&] (const vector<const Block*>& from, const TShape& from_shape,
             Block* to, const TShape& to_shape) {
          // Concat function.
          vector<NodeEntry> ent;
          for (auto blk : from) {
            ent.push_back(blk->entry);
          }
          to->entry = ConcatEntry(ent,
                                  to_shape, "var",
                                  i /* dim */,
                                  default_group_ /* device */);
        });
      CHECK(poped.first.type == Scheme::kCut);
      CHECK(poped.first.dim == i);
      CHECK(poped.second == ncuts_per_dim[i]);
    }
    CHECK_EQ(to_grid.num_blocks().Size(), 1); // The result should be only one entry.
    return to_grid.BlockAt(0).entry;
  }
}

Graph GraphPartitioner::Run() {
  // TODO(minjie):
  // - Control dependencies
  // - NodeEntry versions
  // - Node version
  
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  CHECK_EQ(shapes.size(), graph.num_node_entries());
  // Partitioned grid of each entry in the original graph.
  vector<Grid> entry_grids;
  entry_grids.reserve(graph.num_node_entries());
  // Input/Output grids of each operator.
  vector<vector<Grid>> op_input_grids, op_output_grids;
  op_input_grids.resize(graph.num_nodes());
  op_output_grids.resize(graph.num_nodes());
  // Partitioned operators.
  vector<vector<NodePtr>> splitted_nodes;
  splitted_nodes.resize(graph.num_nodes());

  // Construct grids for each node entry.
  for (uint32_t entid = 0; entid < graph.num_node_entries(); ++entid) {
    //LOG(INFO) << "Split entry#" << entid;
    const TShape& shape = shapes[entid];
    const vector<Scheme>& schemes = tiling_.GetEntrySchemes(entid);
    entry_grids.emplace_back(shape, schemes);
  }

  // Construct grids for operator's input/output. Construct splitted operator.
  DFSVisit( src_graph_->outputs, [&](const NodePtr& node) {
    const uint32_t nodeid = graph.node_id(node.get());
    //LOG(INFO) << "Process node#" << nodeid << ": " << node->attrs.name;
    if (node->is_variable()) {
      // Variable node does not have input/output grid because it is always
      // aligned. Split node will be created to dispatch the data to different devices.
      const uint32_t out_ent_id = graph.entry_id(nodeid, 0);
      const NodeEntry out_ent{node, 0, 0};
      CHECK(node_output_shapes_[node].empty());
      node_output_shapes_[node].push_back(shapes[out_ent_id]);
      AssignDefaultGroup(node);  // The original node is assigned to the default group.
      SplitVariableGrid(shapes[out_ent_id], out_ent, &entry_grids[out_ent_id]);
      // TODO: version ?
      return;
    }
    const vector<SchemeRequest>& allreqs = tiling_.GetSchemeRequests(nodeid);
    const vector<size_t>& chosen = tiling_.GetChosenSchemeRequests(nodeid);
    const size_t num_inputs = allreqs[0].input_schemes.size();
    const size_t num_outputs = allreqs[0].output_schemes.size();
    vector<vector<Scheme>> input_schemes(num_inputs), output_schemes(num_outputs);
    for (size_t choseid : chosen) {
      for (size_t i = 0; i < num_inputs; ++i) {
        input_schemes[i].push_back(allreqs[choseid].input_schemes[i]);
      }
      for (size_t i = 0; i < num_outputs; ++i) {
        output_schemes[i].push_back(allreqs[choseid].output_schemes[i]);
      }
    }
    vector<Grid> input_grids, output_grids;
    CHECK_EQ(node->inputs.size(), num_inputs);
    CHECK_EQ(node->num_outputs(), num_outputs);
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const uint32_t in_ent_id = graph.entry_id(node->inputs[i]);
      const TShape& shape = shapes[in_ent_id];
      input_grids.emplace_back(shape, input_schemes[i]);
    }
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_ent_id = graph.entry_id(nodeid, i);
      const TShape& shape = shapes[out_ent_id];
      output_grids.emplace_back(shape, output_schemes[i]);
    }
    op_input_grids[nodeid].swap(input_grids);
    op_output_grids[nodeid].swap(output_grids);

    // Split attributes.
    NodeAttrs attrs = node->attrs;
    attrs.parsed.clear();  // Require attributes to be re-parsed.
    for (size_t choseid : chosen) {
      const SchemeRequest& req = allreqs[choseid];
      CHECK(req.partitioner);
      attrs = req.partitioner(attrs, 2);
    }
    // Create splitted nodes.
    for (size_t i = 0; i < op_output_grids[nodeid][0].TotalNumBlocks(); ++i) {
      NodePtr n = Node::Create();
      n->attrs = attrs;
      n->attrs.name = node->attrs.name + "_" + std::to_string(i);
      // Control dependencies.
      for (const NodeEntry& in_ent : node->inputs) {
        if (in_ent.node->is_variable()) {
           continue;
        }
        const uint32_t in_node_id = graph.node_id(in_ent.node.get());
        CHECK(splitted_nodes[in_node_id].size() > 0);
        n->control_deps.push_back(splitted_nodes[in_node_id][i]);
      }
      // TODO(minjie): Original control dependencies are ignored.
      for (NodePtr depend_node : node->control_deps) {
        const uint32_t depend_nid = graph.node_id(depend_node.get());
        CHECK_LT(depend_nid, nodeid);
        n->control_deps.push_back(splitted_nodes[depend_nid][i]);
        //LOG(INFO) << "Ignored control dep from " << node->attrs.name << " -> " << depend_node->attrs.name;
      }
      AssignDevice(n, i);
      FinalizeNodeCreation(n);
      splitted_nodes[nodeid].push_back(n);
      // Output shapes.
      CHECK(node_output_shapes_[n].empty());
      for (size_t outidx = 0; outidx < op_output_grids[nodeid].size(); ++outidx) {
        node_output_shapes_[n].push_back(op_output_grids[nodeid][outidx].block_shape());
      }
    }
  });
    
  // Connect splitted operator to form new graph.
  DFSVisit(src_graph_->outputs, [&](const NodePtr& node) {
    //LOG(INFO) << "Processing Node: " << node->attrs.name;
    const uint32_t nodeid = graph.node_id(node.get());
    if (node->is_variable()) {
      // For variable node. Nothing should be done.
      return;
    }
    // Convert input grids.
    vector<const Grid*> aligned_ingrid(node->inputs.size());
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const uint32_t in_ent_id = graph.entry_id(node->inputs[i]);
      const Grid& ingrid = entry_grids[in_ent_id];
      Grid& aligned = op_input_grids[nodeid][i];
      //LOG(INFO) << "\tConvert input #" << i;
      ConvertGrid(ingrid, &aligned);
      aligned_ingrid[i] = &aligned;
    }
    vector<Grid*> outgrid(node->num_outputs());
    vector<Grid*> aligned_outgrid(node->num_outputs());
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_ent_id = graph.entry_id(nodeid, i);
      outgrid[i] = &entry_grids[out_ent_id];
      aligned_outgrid[i] = &op_output_grids[nodeid][i];
    }

    //LOG(INFO) << "\tPerform op";
    PerformOp(aligned_ingrid, aligned_outgrid, splitted_nodes[nodeid]);

    // Convert output grids.
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      //LOG(INFO) << "\tConvert output #" << i;
      ConvertGrid(*aligned_outgrid[i], outgrid[i]);
    }
  });

  // Final graph.
  Graph ret;
  for (const NodeEntry& out_ent : src_graph_->outputs) {
    // TODO(minjie): For output entries, we adopt similar idea of input ones.
    // Currently we use control dependency to simulate the computation while
    // saving the copy from multiple gpus to cpu.
    const uint32_t entid = graph.entry_id(out_ent);
    ret.outputs.push_back(ConcatVariableGrid(out_ent, entry_grids[entid]));
  }
  const IndexedGraph& retgraph = ret.indexed_graph();
  LOG(INFO) << "Original Graph: #Nodes=" << graph.num_nodes()
            << " #Entries=" << graph.num_node_entries();
  LOG(INFO) << "Partitioned Graph: #Nodes=" << retgraph.num_nodes()
            << " #Entries=" << retgraph.num_node_entries();

  // Shape information.
  ShapeVector new_shapes(retgraph.num_node_entries());
  DFSVisit(ret.outputs, [&] (const NodePtr& node) {
    const uint32_t nodeid = retgraph.node_id(node.get());
    //LOG(INFO) << "Node #" << nodeid << ": " << node->attrs.name;
    CHECK_EQ(node_output_shapes_.at(node).size(), node->num_outputs())
      << node_output_shapes_.at(node).size() << " " << node->num_outputs();
    for (size_t idx = 0; idx < node->num_outputs(); ++idx) {
      const uint32_t entid = retgraph.entry_id(nodeid, idx);
      CHECK_LT(entid, retgraph.num_node_entries());
      new_shapes[entid] = std::move(node_output_shapes_[node][idx]);
    }
  });
  /*for (uint32_t entid = 0; entid < retgraph.num_node_entries(); ++entid) {
    LOG(INFO) << "Entry #" << entid << ": " << new_shapes[entid];
  }*/

  // DType information.
  // TODO: currently make all dtype to be float32.
  DTypeVector new_dtypes(retgraph.num_node_entries(), 0);

  // Device information.
  /*DFSVisit(ret.outputs, [&](const NodePtr& node) {
    if (node->attrs.dict.count("ctx_group") != 0) {
      LOG(INFO) << node->attrs.name << " on device: " << node->attrs.dict.at("ctx_group");
    } else {
      LOG(INFO) << node->attrs.name << " on device: unknown";
    }
  });*/

  ret.attrs["shape"] = std::make_shared<any>(std::move(new_shapes));
  ret.attrs["dtype"] = std::make_shared<any>(std::move(new_dtypes));

  /*cout << "digraph {" << endl;
  const auto& retidx = ret.indexed_graph();
  for (uint32_t nid = 0; nid < retidx.num_nodes(); ++nid) {
    const auto& n = retidx[nid];
    for (const auto& in : n.inputs) {
      cout << "\tn" << in.node_id << "_" << retidx[in.node_id].source->attrs.name
           << " -> n" << nid << "_" << n.source->attrs.name << endl;
    }
  }
  cout << "}" << endl;*/

  return ret;
}

}  // namespace pass
}  // namespace nnvm
