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

namespace nnvm {
namespace pass {

class CutAlgorithm;

// TODO(minjie): how to deal with value overflow?
typedef int64_t cost_t;

class NodeEntryGroups {
  // Some NodeEntrys should have same partition schemes. They should be put in one group.
 public:
  // "equals" is a map from NodeEntryId to NodeEntryId indicating the two nodes should be grouped
  // together. NodeEntry without any groups could be missing in the map, and they will be put in
  // a group that has only one node entry.
  NodeEntryGroups(uint32_t num_node_entries,
                  const std::unordered_map<uint32_t, uint32_t>& equals);

  const std::unordered_set<uint32_t>& operator[](uint32_t group_id) const {
    return groups_[group_id];
  }
  uint32_t group_id(uint32_t entry_id) const {
    return entry2group_.at(entry_id);
  }

 private:
  // Each group is a set of NodeEntryId.
  std::vector<std::unordered_set<uint32_t>> groups_;
  // Map from NodeEntryId to NodeEntryGroupId.
  std::unordered_map<uint32_t, uint32_t> entry2group_;
};

class Levels {
 public:
  // Pair: (levelid, index_within_level).
  typedef std::pair<uint32_t, uint32_t> Index;

  Levels(const NodeEntryGroups* groups): entry_groups_(groups) {}

  inline size_t NumNodeLevels() const {
    return node_levels_.size();
  }

  inline size_t NumEntryGroupLevels() const {
    return entry_group_levels_.size();
  }

  inline const std::vector<uint32_t>& GetNodeLevel(size_t lvl) const {
    return node_levels_[lvl];
  }

  inline const std::vector<uint32_t>& GetEntryGroupLevel(size_t lvl) const {
    return entry_group_levels_[lvl];
  }

  inline Index GetNodeIndex(uint32_t nodeid) const {
    return node2index_.at(nodeid);
  }

  inline Index GetNodeEntryIndex(uint32_t entry_id) const {
    return entry2index_.at(entry_id);
  }

  virtual bool AllowCrossLevelEdges() const = 0;

 protected:
  void AddNode(uint32_t levelid, uint32_t nodeid);

  void AddNodeEntry(uint32_t levelid, uint32_t entry_id);

  // Pointer to the node entry groups (no ownership).
  const NodeEntryGroups* entry_groups_;

  std::vector<std::vector<uint32_t>> node_levels_;
  std::vector<std::vector<uint32_t>> entry_group_levels_;

  std::unordered_map<uint32_t, Levels::Index> node2index_;
  std::unordered_map<uint32_t, Levels::Index> entry2index_;
};

class BFS : public Levels {
  // The stored nodes and entries are represented by ids in IndexedGraph.
  // Note: This BFS does not consider control dependencies between nodes.
 public:
  // Constructor.
  BFS(Graph* src, const NodeEntryGroups* groups);

  // Run BFS from the given start node. Treat graph as undirected one.
  void Run(uint32_t start_node_id);

  // Print graph in a readable way.
  void Print() const;

  inline bool AllowCrossLevelEdges() const override { return false; }
  
 private:
  // Pointer to the source graph (no ownership).
  Graph* src_graph_;

  // Entry to all its producer/consumer nodes.
  std::vector<std::unordered_set<uint32_t>> entry_to_nodes_;
  // Node to all its input/output nodes.
  std::vector<std::unordered_set<uint32_t>> node_to_entries_;
};

class NeuralLevels : public Levels {
  // Construct node and entry levels by the neural network layers.
 public:
  // Constructor.
  NeuralLevels(Graph* src, const NodeEntryGroups* groups);

  void Run();

  // Print graph in a readable way.
  void Print() const;

  inline bool AllowCrossLevelEdges() const override { return true; }

 private:
  // Pointer to the source graph (no ownership).
  Graph* src_graph_;

  // Nodes that have the same name prefix (but not start with "_") will be put in the same group.
  // Nodes from the same group will be put in the same level.
  std::vector<std::vector<uint32_t>> node_groups_;
  std::vector<size_t> nodeid2group_;
  std::vector<size_t> group_topo_order_;

  // Entry to all its producer/consumer nodes.
  std::vector<std::unordered_set<uint32_t>> entry_to_nodes_;
  // Node to all its input/output nodes.
  std::vector<std::unordered_set<uint32_t>> node_to_entries_;

};

class Region {
 public:
  // Constructors.
  Region() {}
  Region(const TShape& shp):
    entry_shape_(shp), region_offset_(shp.ndim()), region_shape_(shp) {
    for (size_t i = 0; i < shp.ndim(); ++i) {
      region_offset_[i] = 0;
    }
  }
  Region(const TShape& ent_shp, const TShape& reg_off,
         const TShape& reg_shp):
    entry_shape_(ent_shp), region_offset_(reg_off),
    region_shape_(reg_shp) {}

  inline const TShape& shape() const { return region_shape_; }

  inline const TShape& offset() const { return region_offset_; }

  inline const TShape& entry_shape() const { return entry_shape_; }

  // Partition this region into two sub-regions.
  std::pair<Region, Region> Split2(const Scheme& sch) const;

  // Return true if the region could be splitted using the given scheme.
  bool CanSplit2(const Scheme& sch) const;

  // Area of the region.
  inline cost_t Area() const { return region_shape_.Size(); }

  // Compute the intersection area.
  static cost_t IntersectArea(const Region& r1, const Region& r2);

  // Compute the conversion cost from r1 to r2. The scheme only
  // partitions regions into two parts.
  static cost_t ConvertCost2(const Region& r1, const Scheme& sch1,
                             const Region& r2, const Scheme& sch2);

 private:
  // Shape of the entry this region belongs to.
  TShape entry_shape_;
  // Region offset, and shape.
  TShape region_offset_, region_shape_;
};

struct DPEntry {
  uint32_t entry_group_id;
  Region region;
  // Recorded optimal schemes for each one-cut algorithm.
  std::vector<Scheme> chosen_schemes;
};

struct DPOp {
  uint32_t node_id;
  std::vector<Levels::Index> input_entry_index;
  std::vector<Levels::Index> output_entry_index;
  std::vector<SchemeRequest> aligned_requests;
  std::vector<Region> input_ghost_regions;
  std::vector<Region> output_ghost_regions;
  // Recorded optimal request for each one-cut algorithm.
  std::vector<size_t> chosen_aligned_requests;
};

struct DPState {
  // Entry schemes represented by this state.
  std::vector<Scheme> schemes;
  // Minimal cost to reach this state.
  cost_t cost = 0;
  // The state index in the previous BFS level that is used to reach the optimal
  // cost of this state.
  int prev_state_index = -1;
  // Aligned request chosen for each operator in this state to get the minimal cost.
  std::vector<size_t> chosen_aligned_requests;

  explicit DPState(const std::vector<Scheme>& schemes): schemes(schemes) {}
};

class Tiling {
 public:
  virtual ~Tiling() = default;
  // Get schemes of a node entry.
  virtual const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const = 0;
  // Get scheme requests of the given node.
  virtual const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const = 0;
  // Get scheme requests chosen for the given node.
  virtual const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const = 0;
};

class MergeTiling : public Tiling {
 public:
  MergeTiling(Graph* src, Tiling* t1, Tiling* t2): t1_(t1), t2_(t2) {
    const IndexedGraph& idxgraph = src->indexed_graph();
    entry_schemes_.resize(idxgraph.num_node_entries());
    scheme_requests_.resize(idxgraph.num_nodes());
    chosen_scheme_requests_.resize(idxgraph.num_nodes());
    for (uint32_t entry_id = 0; entry_id < idxgraph.num_node_entries(); ++entry_id) {
      entry_schemes_[entry_id] = t1->GetEntrySchemes(entry_id);
      const auto& from_t2 = t2->GetEntrySchemes(entry_id);
      entry_schemes_[entry_id].insert(
          entry_schemes_[entry_id].end(), from_t2.begin(), from_t2.end());
    }
    for (uint32_t node_id = 0; node_id < idxgraph.num_nodes(); ++node_id) {
      {
      scheme_requests_[node_id] = t1->GetSchemeRequests(node_id);
      const auto& from_t2 = t2->GetSchemeRequests(node_id);
      scheme_requests_[node_id].insert(
          scheme_requests_[node_id].end(), from_t2.begin(), from_t2.end());
      }
      {
      chosen_scheme_requests_[node_id] = t1->GetChosenSchemeRequests(node_id);
      const auto& from_t2 = t2->GetChosenSchemeRequests(node_id);
      chosen_scheme_requests_[node_id].insert(
          chosen_scheme_requests_[node_id].end(), from_t2.begin(), from_t2.end());
      }
    }
  }
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const;
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const {
    return scheme_requests_.at(node_id);
  }
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const {
    return chosen_scheme_requests_.at(node_id);
  }
 private:
  Tiling *t1_, *t2_;

  std::vector<std::vector<Scheme>> entry_schemes_;
  std::vector<std::vector<SchemeRequest>> scheme_requests_;
  std::vector<std::vector<size_t>> chosen_scheme_requests_;
};

class ManualTiling : public Tiling {
 public:
  ManualTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const override {
    return chosen_scheme_requests_.at(node_id);
  }
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const override {
    return aligned_scheme_requests_.at(node_id);
  }
 protected:
  void ChooseSchemeRequests();
  Graph* src_graph_;
  const NodeEntryGroups& entry_groups_;
  const size_t num_devices_;
  const size_t num_cuts_;

  std::vector<std::vector<SchemeRequest>> aligned_scheme_requests_;
  std::vector<std::vector<size_t>> chosen_scheme_requests_;
};

class DataParallelism : public ManualTiling {
 public:
  DataParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const override;
  
 private:
  std::vector<Scheme> param_schemes_;
  std::vector<Scheme> other_schemes_;
  std::vector<std::vector<Scheme>*> entry_schemes_;
};

class ModelParallelism : public ManualTiling {
 public:
  ModelParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const override;

 private:
  std::vector<Scheme> param_schemes_;
  std::vector<Scheme> activation_schemes_;
  std::vector<Scheme> other_schemes_;
  std::vector<std::vector<Scheme>*> entry_schemes_;
};

class CutAlgorithm : public Tiling {
 public:
  // Constructor.
  CutAlgorithm(Graph* src, const Levels& levels,
               const NodeEntryGroups& groups);

  // One cut algorithm. Return the minimal cost.
  cost_t OneCut();

  // K-cut algorithm. Return the minimal cost.
  cost_t KCuts(uint32_t K);

  // Get schemes of a node entry.
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const override;
  
  // Get scheme requests of the given node.
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const override;

  // Get scheme requests chosen for the given node.
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const override;

  // Print debug information.
  void Print() const;

 private:
  // Init all DP states. Create auxiliary structures for the main algorithm.
  void Init();
  
  // Clear all states computed by DP, but leave those auxiliary structures.
  void Reset();

  inline bool IsVariable(const DPOp& op) const {
    return src_graph_->indexed_graph()[op.node_id].source->is_variable();
  }

  std::pair<cost_t, size_t> ConversionCost(
      const DPOp& op,
      const DPState* prev_state,
      const DPState* next_state,
      size_t lvl) const;

  // Extract optimal plan from the states computed by one-cut algorithm. The plan includes
  // schemes of each node entry and which aligned request is used for each node.
  // Return the minimal cost.
  cost_t ExtractOptimalPlan();

  Graph* src_graph_;
  const Levels& levels_;
  const NodeEntryGroups& entry_groups_;

  std::vector<std::vector<DPOp>> dp_operators_;
  std::vector<std::vector<DPEntry>> dp_entries_;

  std::vector<std::vector<DPState>> dp_states_;
};

class GraphPartitioner {
 public:
  GraphPartitioner(const Tiling& tiling, Graph* src,
      const std::string& comm_name, size_t num_devices):
    tiling_(tiling), src_graph_(src), num_devices_(num_devices) {
    comm_planner_ = CommPlanner::CreatePlanner(comm_name);
  }

  void SetOversharding(bool flag) {
    oversharding_ = flag;
  }

  void SetDefaultGraph(const std::string& group) {
    default_group_ = group;
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
  std::string default_group_;

  std::unordered_map<NodePtr, std::vector<TShape>> node_output_shapes_;
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_PARTITION_H_
