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
  os << "] nodes=[";
  gv.DFSNodeVisit([&] (const Node* node) {
        os << node->attrs.name << " ";
      });
  os << "] entries=[";
  gv.DFSEntryVisit([&] (const Node* node, vector<uint32_t> oidx) {
        for (uint32_t i : oidx) {
          os << node->attrs.name << "#" << i << " ";
        }
      }, GraphView::VisitOption::kNoStart);
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
  LOG(FATAL) << "Not implemented!";
  return false;
}

void GraphView::Merge(const GraphView& other) {
  CHECK(this->graph_ == other.graph_);
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
  const auto& conn = graph_->connectivity();
  start_set_.clear();
  end_set_.clear();
  for (const auto& e : start_entries_) start_set_.insert(idx.entry_id(e));
  for (const auto& e : end_entries_) end_set_.insert(idx.entry_id(e));

  node_ids_.clear();
  node_hashes_.clear();
  unordered_map<const Node*, unordered_set<uint32_t>> node2idx;

  vector<const Node*> heads;
  unordered_set<const Node*> roots;
  for (const auto& e : end_entries_) {
    const Node* n = idx[e.node_id].source;
    heads.push_back(n);
    node2idx[n].insert(e.index);
  }
  for (const auto& e : start_entries_) {
    const Node* n = idx[e.node_id].source;
    roots.insert(n);
    node2idx[n].insert(e.index);
  }
  DFSVisitWithRoot(heads, roots,
      [&] (const Node* node, unordered_set<uint32_t> visited_oidx) {
        /*uint32_t nid = idx.node_id(node);
        vector<uint32_t> odix;
        for (uint32_t i = 0; i < node->num_outputs(); ++i) {
          for (const auto& e : end_entries_) {
            if ((nid == e.node_id && i == e.index)
                || conn.has_path(node, i, idx[e.node_id].source)) {
              oidx.push_back(i);
              break;
            }
          }
        }*/
        for (uint32_t i : visited_oidx) {
          //LOG(INFO) << "-------> " << node->attrs.name << "#" << i;
          node2idx[node].insert(i);
        }
      });
  for (const auto& kv : node2idx) {
    const uint32_t nid = idx.node_id(kv.first);
    vector<uint32_t> oidx(kv.second.begin(), kv.second.end());
    std::sort(oidx.begin(), oidx.end());
    node_ids_.push_back(std::make_pair(nid, oidx));
    node_hashes_.insert(nid);
  }
  std::sort(node_ids_.begin(), node_ids_.end(),
      [] (const pair<uint32_t, vector<uint32_t>>& n1,
          const pair<uint32_t, vector<uint32_t>>& n2)->bool {
        return n1.first < n2.first;
      });
}

}  // namespace nnvm
