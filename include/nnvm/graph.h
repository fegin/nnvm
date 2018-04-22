/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nnvm as well as basic data structure.
 */
#ifndef NNVM_GRAPH_H_
#define NNVM_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "./base.h"
#include "./node.h"
#include "./symbolic.h"

namespace nnvm {

class IndexedGraph;

class Connectivity;

/*!
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph {
 public:
  /*! \brief outputs of the computation graph. */
  std::vector<NodeEntry> outputs;
  /*!
   * \brief attributes of a graph
   *  Note that attribute is shared pointer and can be shared across graphs.
   *
   *  It is highly recommended to keep each attribute immutable.
   *  It is also safe to implement an copy-on-write semnatics.
   *
   *  Copy when shared_ptr.unique is not true, while reuse original space
   *  when shared_ptr.unique is true.
   */
  std::unordered_map<std::string, std::shared_ptr<any> > attrs;
  /*!
   * \brief Get the immutable attribute from attrs.
   * \param attr_name the name of the attribute
   * \return the reference to corresponding attribute
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline const T& GetAttr(const std::string& attr_name) const;
  /*!
   * \brief Get a move copy of the attribute, implement copy on write semantics.
   *  The content is moved if the reference counter of shared_ptr is 1.
   *  The attribute is erased from attrs after the call.
   *
   * \param attr_name the name of the attribute
   * \return a new copy of the corresponding attribute.
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline T MoveCopyAttr(const std::string& attr_name);
  /*!
   * \brief get a indexed graph of current graph, if not exist, create it on demand
   * \return The indexed graph.
   * \sa IndexedGraph
   */
  const IndexedGraph& indexed_graph() const;

  const Connectivity& connectivity() const;

 private:
  // internal structure of indexed graph
  mutable std::shared_ptr<const IndexedGraph> indexed_graph_;

  mutable std::shared_ptr<const Connectivity> connectivity_;
};

/*!
 * \brief Auxililary data structure to index a graph.
 *  It maps Nodes in the graph to consecutive integers node_id.
 *  It also maps IndexedGraph::NodeEntry to consecutive integer entry_id.
 *  This allows storing properties of Node and NodeEntry into
 *  compact vector and quickly access them without resorting to hashmap.
 *
 *  The node_id and entry_rptr are the same as the JSON graph produced by SaveJSON Pass.
 */
class IndexedGraph {
 public:
  /*! \brief represents a data in the graph */
  struct NodeEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t node_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief version of the node */
    uint32_t version;
  };
  /*! \brief Node data structure in IndexedGraph */
  struct Node {
    /*! \brief pointer to the source node */
    const nnvm::Node* source;
    /*! \brief inputs to the node */
    array_view<NodeEntry> inputs;
    /*! \brief control flow dependencies to the node */
    array_view<uint32_t> control_deps;
  };
  /*! \return number of nodes in the graph */
  inline size_t num_nodes() const {
    return nodes_.size();
  }
  /*! \return total number of NodeEntry in the graph */
  inline size_t num_node_entries() const {
    return entry_rptr_.back();
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param node_id The node index
   * \param index the output index
   * \return the unique index.
   */
  inline uint32_t entry_id(uint32_t node_id, uint32_t index) const {
    return entry_rptr_[node_id] + index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const NodeEntry& e) const {
    return entry_rptr_[e.node_id] + e.index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given NodeEntry.
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const nnvm::NodeEntry& e) const {
    return entry_rptr_[node_id(e.node.get())] + e.index;
  }
  /*!
   * \brief Get the corresponding node id for a given Node in the IndexedGraph.
   * \param node The Node to query for index.
   * \return the node index.
   */
  inline uint32_t node_id(const nnvm::Node* node) const {
    return node2index_.at(node);
  }
  /*!
   * \brief Return true if the node pointer exists in the graph.
   * \param node The Node pointer.
   * \return true if the node pointer exists in the graph.
   */
  inline bool has_node(const nnvm::Node* node) const {
    return node2index_.find(node) != node2index_.end();
  }
  /*!
   * \brief Get the corresponding Node structure for a given node_id.
   * \param node_id The node id
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](uint32_t node_id) const {
    return nodes_[node_id];
  }
  /*!
   * \brief Get the corresponding Node structure
   * \param node The pointer to the Node structure
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](const nnvm::Node* node) const {
    return nodes_[node_id(node)];
  }
  // convert to indexed node entry
  inline NodeEntry get_index_entry(const nnvm::NodeEntry& ent) const {
    return NodeEntry{node_id(ent.node.get()), ent.index, ent.version};
  }
  /*! \return list of argument nodes */
  inline const std::vector<uint32_t>& input_nodes() const {
    return input_nodes_;
  }
  /*! \return list of mutable nodes */
  inline const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return mutable_input_nodes_;
  }
  /*! \return list of output entries */
  inline const std::vector<NodeEntry>& outputs() const {
    return outputs_;
  }
  // disalllow copy assign
  IndexedGraph(const IndexedGraph&) = delete;

 private:
  friend class Graph;
  /*!
   * \brief Constructor an IndexedGraph from normal Graph
   * \param other The source graph.
   */
  explicit IndexedGraph(const Graph& other);
  // Node pointers in CSR structure.
  std::vector<Node> nodes_;
  // Index to all input nodes.
  std::vector<uint32_t> input_nodes_;
  // Index to all mutable input nodes.
  std::unordered_set<uint32_t> mutable_input_nodes_;
  // space to store the outputs entries
  std::vector<NodeEntry> outputs_;
  // mapping from node to index.
  std::unordered_map<const nnvm::Node*, uint32_t> node2index_;
  // CSR pointer of node entries
  std::vector<size_t> entry_rptr_;
  // space to store input entries of each
  std::vector<NodeEntry> input_entries_;
  // control flow dependencies
  std::vector<uint32_t> control_deps_;
};

inline bool operator == (const IndexedGraph::NodeEntry& e1, const IndexedGraph::NodeEntry& e2) {
  return (e1.node_id == e2.node_id) && (e1.index == e2.index) && (e1.version == e2.version);
}
inline bool operator != (const IndexedGraph::NodeEntry& e1, const IndexedGraph::NodeEntry& e2) {
  return !(e1 == e2);
}
inline bool operator < (const IndexedGraph::NodeEntry& e1, const IndexedGraph::NodeEntry& e2) {
  return std::tie(e1.node_id, e1.index, e1.version) < std::tie(e2.node_id, e2.index, e2.version);
}
} // namespace nnvm

namespace std {
template <>
struct hash<nnvm::IndexedGraph::NodeEntry> {
  size_t operator () (const nnvm::IndexedGraph::NodeEntry& e) const {
    return std::hash<size_t>()(e.node_id) ^
          (std::hash<size_t>()(e.index) << 1 >> 1) ^
          (std::hash<size_t>()(e.version) << 1);
  }
};
}  // namespace std

namespace nnvm {
class Connectivity {
 public:
  bool has_path(const Node* n1, const Node* n2) const {
    return node_predecessors_.at(n2).count(n1);
  }

  bool has_path(const Node* n1, uint32_t idx,
                const Node* n2) const {
    for (const Node* n : entry_consumers_.at(n1)[idx]) {
      if (n == n2 || has_path(n, n2)) {
        return true;
      }
    }
    return false;
  }

  const std::unordered_set<const Node*>& entry_consumers(
      const Node* node, uint32_t idx) const {
    return entry_consumers_.at(node)[idx];
  }

 private:
  friend class Graph;
  explicit Connectivity(const Graph& other);
  std::unordered_map<const Node*, std::unordered_set<const Node*> > node_predecessors_;
  std::unordered_map<const Node*, std::vector<std::unordered_set<const Node*> > > entry_consumers_;
};

/*!
 * \brief perform a Post Order DFS visit to each node in the graph.
 *  This order is deterministic and is also topoligical sorted.
 * \param heads The heads in the graph.
 * \param fvisit a function of type std::function<void(const std::shared_ptr<Node>&)>
 * \tparam FVisit The function type to perform the visit.
 */
template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads, FVisit fvisit);

// inline function implementations
template<typename T>
inline const T& Graph::GetAttr(const std::string& attr_name) const {
  auto it = attrs.find(attr_name);
  CHECK(it != attrs.end())
      << "Cannot find attribute " << attr_name << " in the graph";
  return nnvm::get<T>(*it->second);
}

template<typename T>
inline T Graph::MoveCopyAttr(const std::string& attr_name) {
  auto it = attrs.find(attr_name);
  CHECK(it != attrs.end())
      << "Cannot find attribute " << attr_name << " in the graph";
  std::shared_ptr<any> sptr = it->second;
  attrs.erase(it);
  if (sptr.unique()) {
    return std::move(nnvm::get<T>(*sptr));
  } else {
    return nnvm::get<T>(*sptr);
  }
}

template <typename GNode, typename HashType,
          typename FVisit, typename HashFunc,
          typename InDegree, typename GetInput>
void PostOrderDFSVisit(const std::vector<GNode>& heads,
                       FVisit fvisit,
                       HashFunc hash,
                       InDegree indegree,
                       GetInput getinput) {
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_set<HashType> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(head_hash);
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        fvisit(back.first);
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (visited.count(input_hash) == 0) {
          stack.push_back(std::make_pair(input, 0));
          visited.insert(input_hash);
        }
      }
    }
  }
}

template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads,
                     FVisit fvisit) {
  typedef const NodePtr* GNode;
  std::vector<GNode> head_nodes(heads.size());
  std::transform(heads.begin(), heads.end(), head_nodes.begin(),
                 [](const NodeEntry& e)->GNode {
                   return &e.node;
                 });
  PostOrderDFSVisit<GNode, Node*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->Node* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t {  // InDegree
        return (*n)->inputs.size() + (*n)->control_deps.size();
      },
      [](GNode n, uint32_t index)->GNode {  // GetInput
        if (index < (*n)->inputs.size()) {
          return &(*n)->inputs.at(index).node;
        } else {
          return &(*n)->control_deps.at(index - (*n)->inputs.size());
        }
      });
}

// Traverse the graph from the given root node to the given heads.
// Nodes that can only connect to heads through nodes in root will
// NOT be visited (in other word, those nodes will be disconnected from
// heads if nodes in root are removed from the graph). Nodes in root
// will NOT be visited.
template <typename GNode, typename HashType,
          typename FVisit, typename HashFunc,
          typename InDegree, typename GetInput,
          typename IsRoot>
void PostOrderDFSVisitWithRoot(
    const std::vector<GNode>& heads,
    FVisit fvisit,
    HashFunc hash,
    InDegree indegree,
    GetInput getinput,
    IsRoot isroot) {
  std::unordered_set<HashType> all_head_hash;
  for (auto& head : heads) {
    all_head_hash.insert(hash(head));
  }
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_map<HashType, bool> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0 && !isroot(head)) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(std::make_pair(head_hash, false));
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        if (visited.at(back.first)) {
          fvisit(back.first);
        }
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (all_head_hash.count(input_hash)) {
          // Found head nodes, stop.
        } else if (isroot(input)) {
          // Found root nodes, stop.
          for (auto& t : stack) {
            visited.at(t.first) = true;
          }
        } else if (visited.count(input_hash) == 0) {
          // Continue search.
          stack.push_back(std::make_pair(input, 0));
          visited.insert(std::make_pair(input_hash, false));
        }
      }
    }
  }
}

template<typename FVisit>
inline void DFSVisitWithRoot(
    const std::vector<const Node*>& head_nodes,
    const std::unordered_set<const Node*>& root,
    FVisit fvisit) {
  typedef const Node* GNode;
  PostOrderDFSVisitWithRoot<GNode, const Node*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(n); },  // FVisit
      [](GNode n) { return n; },  // HashFunc
      [](GNode n)->uint32_t {  // InDegree
        return n->inputs.size();
      },
      [](GNode n, uint32_t index)->GNode {  // GetInput
        return n->inputs.at(index).node.get();
      },
      [&root](GNode n)->bool { return root.count(n); });
}

/* A graph view represents all nodes and entries that
 * are reacheable from start entries substract those
 * are reachable from end enries.*/
class GraphView {
  friend std::ostream& operator << (std::ostream& os, const GraphView& region);
 public:
  GraphView(Graph* g,
      const std::vector<IndexedGraph::NodeEntry>& st,
      const std::vector<IndexedGraph::NodeEntry>& ed):
    graph_(g), start_entries_(st), end_entries_(ed) {
    BuildSet();
  }

  GraphView(Graph* g, const Node* node): graph_(g) {
    const auto& idx = g->indexed_graph();
    CHECK(idx.has_node(node));
    CHECK(node->num_outputs() != 0);
    uint32_t nid = idx.node_id(node);
    if (!node->inputs.empty()) {
      start_entries_.push_back(idx[nid].inputs[0]);
    }
    end_entries_.push_back(IndexedGraph::NodeEntry{nid, 0, 0});
    BuildSet();
  }

  GraphView(Graph* g, const NodeEntry& ent): graph_(g) {
    const auto& idx = g->indexed_graph();
    const auto& e = idx.get_index_entry(ent);
    start_entries_.push_back(e);
    end_entries_.push_back(e);
    BuildSet();
  }

  // A merge is NOT a union of two views.
  void Merge(const GraphView& other);

  bool Contains(const IndexedGraph::NodeEntry& e) const;

  // FVisit is : void(const Node* node, vector<uint32_t> outidx);
  // Note: Both start and end entries will be visited.
  template<typename FVisit>
  void DFSEntryVisit(FVisit fvisit) const;

  // FVisit is : void(const Node* node);
  // Note: Only nodes depending on the start entries will be visited.
  template<typename FVisit>
  void DFSNodeVisit(FVisit fvisit) const {
    const auto& idx = graph_->indexed_graph();
    this->DFSEntryVisit(
        [&] (const Node* node, const std::vector<uint32_t>& outidx) {
          const uint32_t node_id = idx.node_id(node);
          bool flag = true;
          for (uint32_t i : outidx) {
            const uint32_t eid = idx.entry_id(node_id, i);
            if (start_set_.count(eid)) {
              flag = false;
              break;
            }
          }
          if (flag) {
            fvisit(node);
          }
        });
  }

 private:
  void BuildSet();

  Graph* graph_;
  std::vector<IndexedGraph::NodeEntry> start_entries_, end_entries_;
  std::unordered_set<uint32_t> start_set_, end_set_;

  std::vector<uint32_t> node_ids_;
};

std::ostream& operator << (std::ostream& os, const GraphView& gv);

template<typename FVisit>
void GraphView::DFSEntryVisit(FVisit fvisit) const {
  const auto& idx = graph_->indexed_graph();
  const auto& conn = graph_->connectivity();
  std::unordered_set<const Node*> root;
  for (const auto& ent : start_entries_) {
    root.insert(idx[ent.node_id].source);
  }
  std::vector<const Node*> heads;
  std::unordered_map<const Node*, std::vector<uint32_t> > head_ent_idx;
  for (const auto& ent : end_entries_) {
    const Node* node = idx[ent.node_id].source;
    heads.push_back(node);
    head_ent_idx[node].push_back(ent.index);
  }
  DFSVisitWithRoot(heads, root,
    [&] (const Node* node) {
      const uint32_t nid = idx.node_id(node);
      if (head_ent_idx.count(node)) {
        fvisit(node, head_ent_idx.at(node));
        return;
      }
      // Prune out nodes that have no path from start entries.
      std::vector<uint32_t> outidx;
      for (uint32_t i = 0; i < node->num_outputs(); ++i) {
        IndexedGraph::NodeEntry ent{nid, i, 0};
        if (this->Contains(ent)) {
          outidx.push_back(i);
        }
      }
      if (!outidx.empty()) {
        fvisit(node, outidx);
      }
    });
}

}  // namespace nnvm

#endif  // NNVM_GRAPH_H_
