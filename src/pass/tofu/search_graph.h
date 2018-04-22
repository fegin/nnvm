/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file geometry.h
 * \brief Graph structure used by search algorithm.
 */
#ifndef NNVM_PASS_SEARCH_GRAPH_H_
#define NNVM_PASS_SEARCH_GRAPH_H_

#include <nnvm/base.h>
#include <nnvm/scheme.h>
#include <nnvm/graph.h>
#include <unordered_set>

namespace nnvm {
namespace pass {

class NodeEntryGroups {
  // Some NodeEntrys should have same partition schemes. They should be put in one group.
 public:
  // "equals" is a map from NodeEntryId to NodeEntryId indicating the two nodes should be grouped
  // together. NodeEntry without any groups could be missing in the map, and they will be put in
  // a group that has only one node entry.
  NodeEntryGroups(uint32_t num_node_entries,
                  const std::vector<std::pair<uint32_t, uint32_t>>& equals);

  const std::unordered_set<uint32_t>& operator[](uint32_t group_id) const {
    return groups_[group_id];
  }
  uint32_t group_id(uint32_t entry_id) const {
    return entry2group_[entry_id];
  }

 private:
  // Each group is a set of NodeEntryId.
  std::vector<std::unordered_set<uint32_t>> groups_;
  // Map from NodeEntryId to NodeEntryGroupId.
  std::vector<uint32_t> entry2group_;
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

  void RemoveExtraNodeLevel();

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

class MegaGraph {
 public:
  MegaGraph(Graph* orig);

  void Print();

  void MergeElementwise();

  void MergeWeightAndGradients();

  void MergeRNNSteps();

  const std::vector<std::pair<uint32_t, uint32_t>>& entry_equals() const {
    return entry_equals_;
  }

 private:
  bool IsElementWise(uint32_t nid);

 private:
  enum class MapType {
    kNA,
    kMapToNode,
    kMapToEntry,
  };

  Graph graph_;
  Graph* orig_graph_;

  // array of length num_nodes(orig_graph_)
  std::vector<MapType> orignode_mapping_type_;
  std::vector<nnvm::any> orignode_mappings_;
  // array of length num_node_entries(orig_graph_)
  std::vector<MapType> origentry_mapping_type_;
  std::vector<nnvm::any> origentry_mappings_;

  std::unordered_map<const Node*, uint32_t> num_outputs_;

  // Group of nodes in the original graph. This is a map from the
  // meta graph node_id to its associated node group.
  // The first node in the group is always the forward node in the original graph.
  //std::vector<const Node*> orignode2meganode_;
  std::unordered_map<const Node*, GraphView> node_mappings_;

  // Group of entries in the original graph. This is a map from
  // the meta graph entry_id to its associated entry group.
  std::unordered_map<const Node*, std::vector<GraphView>> entry_mappings_;

  std::vector<std::pair<uint32_t, uint32_t>> entry_equals_;
  std::vector<std::pair<uint32_t, uint32_t>> node_equals_;
};

}  // namespace pass
}  // namespace nnvm

#endif   // NNVM_PASS_SEARCH_GRAPH_H_
