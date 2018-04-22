/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph_attr_types.cc
 * \brief Graph node data structure.
 */
#include <nnvm/graph.h>
#include <nnvm/op_attr_types.h>
#include <limits>
#include <algorithm>
#include <tuple>

using namespace std;

namespace nnvm {

const IndexedGraph& Graph::indexed_graph() const {
  if (indexed_graph_ == nullptr) {
    indexed_graph_.reset(new IndexedGraph(*this));
  }
  return *indexed_graph_;
}

const Connectivity& Graph::connectivity() const {
  if (connectivity_ == nullptr) {
    connectivity_.reset(new Connectivity(*this));
  }
  return *connectivity_;
}

// implement constructor from graph
IndexedGraph::IndexedGraph(const Graph &g) {
  entry_rptr_.push_back(0);
  std::vector<size_t> inputs_rptr{0}, control_rptr{0};

  DFSVisit(g.outputs, [this, &inputs_rptr, &control_rptr]
             (const NodePtr& n) {
      CHECK_LT(nodes_.size(), std::numeric_limits<uint32_t>::max());
      uint32_t nid = static_cast<uint32_t>(nodes_.size());
      // nodes_
      IndexedGraph::Node new_node;
      new_node.source = n.get();
      nodes_.emplace_back(std::move(new_node));
      // arg_nodes_
      if (n->is_variable()) {
        input_nodes_.push_back(nid);
      }
      // node2index_
      node2index_[n.get()] = nid;
      // entry rptr
      entry_rptr_.push_back(entry_rptr_.back() + n->num_outputs());
      // input entries
      for (const auto& e : n->inputs) {
        auto it = node2index_.find(e.node.get());
        CHECK(it != node2index_.end() && it->first == e.node.get());
        input_entries_.emplace_back(NodeEntry{it->second, e.index, e.version});
      }
      inputs_rptr.push_back(input_entries_.size());
      // control deps
      for (const auto& nptr : n->control_deps) {
        auto it = node2index_.find(nptr.get());
        CHECK(it != node2index_.end() && it->first == nptr.get());
        control_deps_.push_back(it->second);
      }
      control_rptr.push_back(control_deps_.size());
  });

  for (const auto& e : g.outputs) {
    outputs_.emplace_back(NodeEntry{
        node2index_.at(e.node.get()), e.index, e.version});
  }

  static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");
  std::unordered_set<uint32_t> mutable_inputs;
  // setup array view
  // input_entries_ and control_rptr must not change after this step.
  const NodeEntry* iptr = dmlc::BeginPtr(input_entries_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].inputs = array_view<NodeEntry>(
        iptr + inputs_rptr[nid], iptr + inputs_rptr[nid + 1]);
    if (nodes_[nid].source->op() != nullptr &&
        fmutate_inputs.count(nodes_[nid].source->op())) {
      for (uint32_t i : fmutate_inputs[nodes_[nid].source->op()](nodes_[nid].source->attrs)) {
        mutable_input_nodes_.insert(nodes_[nid].inputs[i].node_id);
      }
    }
  }
  const uint32_t* cptr = dmlc::BeginPtr(control_deps_);
  for (size_t nid = 0; nid < nodes_.size(); ++nid) {
    nodes_[nid].control_deps = array_view<uint32_t>(
        cptr + control_rptr[nid], cptr + control_rptr[nid + 1]);
  }
}

Connectivity::Connectivity(const Graph& other) {
  const auto& idx = other.indexed_graph();
  DFSVisit(other.outputs, [&] (const NodePtr& node) {
      entry_consumers_[node.get()].resize(node->num_outputs());
      for (const auto& ent : node->inputs) {
        const Node* innode = ent.node.get();
        node_predecessors_[node.get()].insert(innode);
        node_predecessors_[node.get()].insert(
            node_predecessors_[innode].begin(),
            node_predecessors_[innode].end());
        entry_consumers_[innode][ent.index].insert(node.get());
      }
    });
  //for (const auto& kv : node_predecessors_) {
    //std::cout << kv.first->attrs.name << " <- [";
    //for (const auto& x : kv.second) {
      //std::cout << x->attrs.name << " " ;
    //}
    //std::cout << "]" << std::endl;
  //}
}

std::ostream& operator << (std::ostream& os, const GraphView& gv) {
  const auto& idx = gv.graph_->indexed_graph();
  os << "From: [";
  for (const auto& ent : gv.start_entries_) {
    os << idx[ent.node_id].source->attrs.name << "#" << ent.index << " ";
  }
  os << "] To: [";
  for (const auto& ent : gv.end_entries_) {
    os << idx[ent.node_id].source->attrs.name << "#" << ent.index << " ";
  }
  os << "] [";
  for (const uint32_t nid : gv.node_ids_) {
    os << idx[nid].source->attrs.name << " ";
  }
  return os << "]";
}

namespace {
bool ContainsIn(const Graph& graph,
                const std::vector<IndexedGraph::NodeEntry>& st,
                const std::vector<IndexedGraph::NodeEntry>& ed,
                const IndexedGraph::NodeEntry& e) {
  const auto& idx = graph.indexed_graph();
  const auto& conn = graph.connectivity();
  const Node* n = idx[e.node_id].source;
  bool flag = st.empty()? true : false;
  for (const auto& x : st) {
    if (conn.has_path(idx[x.node_id].source, x.index, n)) {
      flag = true;
      break;
    }
  }
  if (!flag) return false;
  flag = false;
  for (const auto& x : ed) {
    if (conn.has_path(n, e.index, idx[x.node_id].source)) {
      flag = true;
      break;
    }
  }
  return flag;

}
}  // namespace

bool GraphView::Contains(const IndexedGraph::NodeEntry& e) const {
  const auto& idx = graph_->indexed_graph();
  const uint32_t eid = idx.entry_id(e);
  if (end_entries_.empty()) return false;
  if (start_set_.count(eid) || end_set_.count(eid)) {
    return true;
  }
  return ContainsIn(*graph_, start_entries_, end_entries_, e)
    && !ContainsIn(*graph_, end_entries_, start_entries_, e);
}

void GraphView::Merge(const GraphView& other) {
  CHECK(this->graph_ == other.graph_);
  /*const auto& idx = graph_->indexed_graph();
  std::vector<IndexedGraph::NodeEntry> new_st, new_ed;
  for (const auto& e : this->start_entries_) {
    const uint32_t eid = idx.entry_id(e);
    if (other.start_set_.count(eid) || !other.Contains(e)) {
      new_st.push_back(e);
    }
  }
  for (const auto& e : other.start_entries_) {
    const uint32_t eid = idx.entry_id(e);
    if (this->start_set_.count(eid) || !this->Contains(e)) {
      new_st.push_back(e);
    }
  }
  for (const auto& e : this->end_entries_) {
    const uint32_t eid = idx.entry_id(e);
    if (other.end_set_.count(eid) || !other.Contains(e)) {
      new_ed.push_back(e);
    }
  }
  for (const auto& e : other.end_entries_) {
    const uint32_t eid = idx.entry_id(e);
    if (this->end_set_.count(eid) || !this->Contains(e)) {
      new_ed.push_back(e);
    }
  }
  start_entries_.swap(new_st);
  end_entries_.swap(new_ed);*/
  start_entries_.insert(start_entries_.end(),
                        other.start_entries_.begin(),
                        other.start_entries_.end());
  end_entries_.insert(end_entries_.end(),
                      other.end_entries_.begin(),
                      other.end_entries_.end());
  BuildSet();
}

void GraphView::BuildSet() {
  std::sort(start_entries_.begin(), start_entries_.end());
  std::sort(end_entries_.begin(), end_entries_.end());
  const auto& idx = graph_->indexed_graph();
  start_set_.clear();
  end_set_.clear();
  for (const auto& e : start_entries_) start_set_.insert(idx.entry_id(e));
  for (const auto& e : end_entries_) end_set_.insert(idx.entry_id(e));

  node_ids_.clear();
  vector<const Node*> heads;
  unordered_set<const Node*> roots;
  for (const auto& e : end_entries_) {
    heads.push_back(idx[e.node_id].source);
  }
  for (const auto& e : start_entries_) {
    roots.insert(idx[e.node_id].source);
  }

  DFSVisitWithRoot(heads, roots,
      [&] (const Node* node) {
        node_ids_.push_back(idx.node_id(node));
      });
}

}  // namespace nnvm
