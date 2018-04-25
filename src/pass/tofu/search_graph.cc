#include "./search_graph.h"

#include <nnvm/graph_attr_types.h>
#include <queue>

#include "./utils.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {
string GetPrefix(const string& str) {
  // TODO(minjie): Very ugly way of getting forward op name.
  string ret;
  size_t pos = str.find_last_of('/');
  if (pos == 0 || pos == string::npos) {
    return str;
  } else {
    return str.substr(0, pos);
  }
}
}  // namespace

NodeEntryGroups::NodeEntryGroups(
    const Graph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& equals)
  : graph_(graph) {
  const auto& idx = graph.indexed_graph();
  uint32_t num_node_entries = idx.num_node_entries();
  groups_.resize(num_node_entries);
  entry2group_.resize(num_node_entries);
  // Union-find
  vector<uint32_t> groups(num_node_entries);
  for (uint32_t eid = 0; eid < num_node_entries; ++eid) {
    groups[eid] = eid;
  }
  for (const auto& eq : equals) {
    uint32_t g1 = eq.first;
    uint32_t g2 = eq.second;
    while (groups[g1] != g1) g1 = groups[g1];
    while (groups[g2] != g2) g2 = groups[g2];
    const uint32_t group = std::min(g1, g2);
    groups[g1] = group;
    groups[g2] = group;
    groups[eq.first] = group;
    groups[eq.second] = group;
  }
  for (uint32_t eid = 0; eid < num_node_entries; ++eid) {
    uint32_t g = groups[eid];
    while (g != groups[g]) g = groups[g];
    entry2group_[eid] = g;
    groups_[g].insert(eid);
  }
}

void NodeEntryGroups::Print() const {
  const auto& idx = graph_.indexed_graph();
  vector<string> ename(idx.num_node_entries());
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    for (uint32_t i = 0; i < node->num_outputs(); ++i) {
      uint32_t eid = idx.entry_id(nid, i);
      ename[eid] = node->attrs.name + "#" + std::to_string(i);
    }
  }
  size_t ngroups = 0;
  for (uint32_t gid = 0; gid < idx.num_node_entries(); ++gid) {
    if (groups_[gid].empty()) continue;
    ++ngroups;
    LOG(INFO) << "Group#" << gid << "[";
    for (const auto& eid : groups_[gid]) {
      LOG(INFO) << "\t" << ename[eid];
    }
    LOG(INFO) << "]";
  }
  LOG(INFO) << "#Entry groups: " << ngroups;
}

NodeGroups::NodeGroups(
    const Graph& graph,
    const std::vector<std::pair<uint32_t, uint32_t>>& equals)
  : graph_(graph) {
  const auto& idx = graph.indexed_graph();
  uint32_t num_nodes = idx.num_nodes();
  groups_.resize(num_nodes);
  node2group_.resize(num_nodes);
  // Union-find
  vector<uint32_t> groups(num_nodes);
  for (uint32_t nid = 0; nid < num_nodes; ++nid) {
    groups[nid] = nid;
  }
  for (const auto& eq : equals) {
    uint32_t g1 = eq.first;
    uint32_t g2 = eq.second;
    while (groups[g1] != g1) g1 = groups[g1];
    while (groups[g2] != g2) g2 = groups[g2];
    const uint32_t group = std::min(g1, g2);
    groups[g1] = group;
    groups[g2] = group;
    groups[eq.first] = group;
    groups[eq.second] = group;
  }
  for (uint32_t nid = 0; nid < num_nodes; ++nid) {
    uint32_t g = groups[nid];
    while (g != groups[g]) g = groups[g];
    node2group_[nid] = g;
    groups_[g].insert(nid);
  }
}

void NodeGroups::Print() const {
  const auto& idx = graph_.indexed_graph();
  size_t ngroups = 0;
  for (uint32_t gid = 0; gid < idx.num_nodes(); ++gid) {
    if (groups_[gid].empty()) continue;
    ++ngroups;
    LOG(INFO) << "Group#" << gid << "[";
    for (const auto& nid : groups_[gid]) {
      LOG(INFO) << "\t" << idx[nid].source->attrs.name;
    }
    LOG(INFO) << "]";
  }
  LOG(INFO) << "#Node groups: " << ngroups;
}

void Levels::AddNode(uint32_t levelid, uint32_t nodeid) {
  CHECK(node2index_.find(nodeid) == node2index_.end())
    << "Node #" << nodeid << " has already been added to level #" << levelid;
  if (levelid >= node_group_levels_.size()) {
    // New level.
    node_group_levels_.push_back(vector<uint32_t>());
  }
  CHECK_LT(levelid, node_group_levels_.size());
  const uint32_t level_index = node_group_levels_[levelid].size();
  const uint32_t node_group_id = node_groups_->group_id(nodeid);
  node_group_levels_[levelid].push_back(node_group_id);
  // For all node in the same group, make its index.
  for (const uint32_t node : (*node_groups_)[node_group_id]) {
    CHECK(!node2index_.count(node))
      << "Node should not be added twice (" << node << ").";
    node2index_[node] = make_pair(levelid, level_index);
  }
}

void Levels::AddNodeEntry(uint32_t levelid, uint32_t entry_id) {
  if (entry2index_.find(entry_id) != entry2index_.end()) {
    // Already been added (from another node in the group).
    return;
  }
  if (levelid >= entry_group_levels_.size()) {
    // New level.
    entry_group_levels_.push_back(vector<uint32_t>());
  }
  CHECK_LT(levelid, entry_group_levels_.size());
  const uint32_t level_index = entry_group_levels_[levelid].size();
  const uint32_t entry_group_id = entry_groups_->group_id(entry_id);
  entry_group_levels_[levelid].push_back(entry_group_id);
  // For all entry in the same group, make its index.
  for (const uint32_t ent : (*entry_groups_)[entry_group_id]) {
    CHECK(entry2index_.find(ent) == entry2index_.end())
      << "Entry should not be added twice (" << ent << ").";
    entry2index_[ent] = make_pair(levelid, level_index);
  }
}

void Levels::RemoveExtraNodeLevel() {
  if (node_group_levels_.size() > entry_group_levels_.size()) {
    // If the last level is a node level, it can be merged
    // with the second last one.
    const size_t old_size = node_group_levels_.size();
    const auto& last_lvl = node_group_levels_[old_size - 1];
    for (const uint32_t gid : last_lvl) {
      const size_t level_idx = node_group_levels_[old_size - 2].size();
      node_group_levels_[old_size - 2].push_back(gid);
      for (const uint32_t nid : (*node_groups_)[gid]) {
        node2index_[nid] = std::make_pair(old_size - 2, level_idx);
      }
    }
    node_group_levels_.resize(old_size - 1);
  }
  CHECK_EQ(node_group_levels_.size(), entry_group_levels_.size());
}

BFS::BFS(Graph* src, const NodeEntryGroups* entry_group, const NodeGroups* node_group):
  Levels(entry_group, node_group), src_graph_(src) {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  for (uint32_t v = 0; v < graph.num_nodes(); ++v) {
    uint32_t ng = node_groups_->group_id(v);
    unordered_set<uint32_t> neigh_eg;
    for (uint32_t node_id : (*node_groups_)[ng]) {
      const auto& node = graph[node_id];
      // For all input entries, put the node in the adj list.
      for (const auto& in_ent : node.inputs) {
        const uint32_t in_ent_id = graph.entry_id(in_ent);
        neigh_eg.insert(entry_groups_->group_id(in_ent_id));
      }
      // For all output entries, put the node in the adj list.
      for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
        const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
        neigh_eg.insert(entry_groups_->group_id(out_ent_id));
      }
    }
    for (uint32_t eg : neigh_eg) {
      eg2ng_[eg].insert(ng);
      ng2eg_[ng].insert(eg);
    }
  }
}

void BFS::Run(std::vector<uint32_t> start_node_id) {
  queue<pair<uint32_t, uint32_t>> queue;  // (level, gid)
  unordered_set<uint32_t> visited_ng, visited_eg;
  for (uint32_t nid : start_node_id) {
    queue.push(make_pair(0, node_groups_->group_id(nid)));
    visited_ng.insert(node_groups_->group_id(nid));
  }
  while (!queue.empty()) {
    uint32_t level = 0, gid = 0;
    tie(level, gid) = queue.front();
    queue.pop();

    if (level % 2 == 0) {
      // This is a Node.
      for (const uint32_t nid : (*node_groups_)[gid]) {
        AddNode(level / 2, nid);
        break;  // Only add one node is enough. The rest will be added automatically.
      }
      // Put all its input/output entrie groups into queue.
      for (const uint32_t eg : ng2eg_[gid]) {
        if (visited_eg.count(eg) == 0) {
          queue.push(make_pair(level + 1, eg));
          visited_eg.insert(eg);
        }
      }
    } else {
      // This is a NodeEntry.
      visited_eg.insert(gid);
      for (const uint32_t eid : (*entry_groups_)[gid]) {
        AddNodeEntry(level / 2, eid);
        break;  // Only add one entry is enough. The rest will be added automatically.
      }
      // Put all its producers/consumers into queue.
      for (const uint32_t ng : eg2ng_[gid]) {
        if (visited_ng.count(ng) == 0) {
          queue.push(make_pair(level + 1, ng));
          visited_ng.insert(ng);
        }
      }
    }
  }
  RemoveExtraNodeLevel();
}

void BFS::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  /*LOG(INFO) << "NodeEntry To Node";
  for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
    ostringstream oss;
    oss << "Entry#" << entid << ": ";
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      oss << nodeid << " ";
    }
    LOG(INFO) << oss.str();
  }*/
  for (size_t i = 0; i < node_group_levels_.size(); ++i) {
    LOG(INFO) << "Level Node: [";
    for (const uint32_t groupid : node_group_levels_[i]) {
      ostringstream oss;
      oss << "\t{";
      for (const uint32_t nodeid : (*node_groups_)[groupid]) {
        const Node* node = graph[nodeid].source;
        oss << "#" << nodeid << ": \"" << node->attrs.name << "\""
                  << (node->is_variable()? "(variable)" : "") << ", ";
      }
      LOG(INFO) << oss.str() << "},";
    }
    LOG(INFO) << "]";
    if (i < entry_group_levels_.size()) {
      LOG(INFO) << "Level NodeEntry: [";
      for (const uint32_t groupid : entry_group_levels_[i]) {
        ostringstream oss;
        oss << "\t{";
        for (const uint32_t entid : (*entry_groups_)[groupid]) {
          oss << "#" << entid << ": " << shapes[entid] << ", ";
        }
        LOG(INFO) << oss.str() << "},";
      }
      LOG(INFO) << "]";
    }
  }
  LOG(INFO) << "#Levels: " << node_group_levels_.size();
}

/*NeuralLevels::NeuralLevels(Graph* src, const NodeEntryGroups* groups):
  Levels(groups), src_graph_(src) {
  // Create node groups.
  const IndexedGraph& graph = src_graph_->indexed_graph();
  nodeid2group_.resize(graph.num_nodes(), -1); // -1 means the node does not belong to any group.
  unordered_map<string, size_t> prefix2group;
  for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
    const string& name = graph[nodeid].source->attrs.name;
    CHECK(name == "" || name.find_first_of('/') == name.find_last_of('/'))
      << "Unsupported node name: \"" << name << "\"";
    if (name == "" || name == "sum_grad") {
      // TODO(minjie): This is an ugly way to ignore non-symbolic operators.
      // These nodes will be put in a group that contains only themselves.
      nodeid2group_[nodeid] = node_groups_.size();
      node_groups_.push_back({nodeid});
      continue;
    }
    const string& prefix = GetPrefix(name);
    if (prefix2group.find(prefix) == prefix2group.end()) {
      // New node group.
      //LOG(INFO) << "Group " << prefix;
      prefix2group[prefix] = node_groups_.size();
      node_groups_.push_back(vector<uint32_t>());
    }
    size_t groupid = prefix2group[prefix];
    nodeid2group_[nodeid] = groupid;
    node_groups_[groupid].push_back(nodeid);
  }
  //for (size_t i = 0; i < node_groups_.size(); ++i) {
  //  LOG(INFO) << "Group #" << i << ": {";
  //  for (uint32_t nodeid : node_groups_[i]) {
  //    LOG(INFO) << "\t#" << nodeid << ": " << graph[nodeid].source->attrs.name << ",";
  //  }
  //  LOG(INFO) << "}";
  //}
  // Following is the same as in BFS. Create undirected graph from original graph.
  entry_to_nodes_.resize(graph.num_node_entries());
  node_to_entries_.resize(graph.num_nodes());
  for (uint32_t node_id = 0; node_id < graph.num_nodes(); ++node_id) {
    const IndexedGraph::Node& node = graph[node_id];
    // For all input entries, put the node in the adj list.
    for (const IndexedGraph::NodeEntry& in_ent : node.inputs) {
      const uint32_t in_ent_id = graph.entry_id(in_ent);
      entry_to_nodes_[in_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(in_ent_id);
    }
    // For all output entries, put the node in the adj list.
    for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
      const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
      entry_to_nodes_[out_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(out_ent_id);
    }
  }
}

void NeuralLevels::Run() {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  // Organize nodes in topological order.
  vector<uint32_t> topo_order;
  DFSVisit(src_graph_->outputs, [&](const NodePtr& node) {
      //LOG(INFO) << "Node #" << graph.node_id(node.get())
                //<< ": " << node->attrs.name;
      topo_order.push_back(graph.node_id(node.get()));
    });
  // Add node group in topo order.
  int curlevel = -1;
  vector<size_t> levelid(graph.num_nodes(), 0);
  vector<int> group2level(node_groups_.size(), -1);
  for (size_t i = 0; i < topo_order.size(); ++i) {
    const uint32_t nodeid = topo_order[i];
    const size_t groupid = nodeid2group_[nodeid];
    if (group2level[groupid] < 0) {
      if (i > 0 && levelid[i - 1] < curlevel) {
        // XXX(minjie): Special treatment for operators not appear in forward pass.
        // This is not a generic rule!
        group2level[groupid] = levelid[i - 1];
      } else {
        ++curlevel;
        group2level[groupid] = curlevel;
      }
    }
    levelid[nodeid] = group2level[groupid];
  }
  //for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
  //    LOG(INFO) << "Node #" << nodeid
  //              << ": " << graph[nodeid].source->attrs.name
  //              << " " << levelid[nodeid];
  //}
  // Make levels.
  for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
    AddNode(levelid[nodeid], nodeid);
  }
  for (uint32_t entid = 0; entid < graph.num_node_entries(); ++entid) {
    // Always add node entry to the smallest levels of its connected nodes.
    size_t entlvl = std::numeric_limits<size_t>::max();
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      entlvl = std::min(entlvl, levelid[nodeid]);
    }
    AddNodeEntry(entlvl, entid);
  }
  RemoveExtraNodeLevel();
}

void NeuralLevels::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  LOG(INFO) << "NodeEntry To Node";
  //for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
  //  ostringstream oss;
  //  oss << "Entry#" << entid << ": ";
  //  for (uint32_t nodeid : entry_to_nodes_[entid]) {
  //    oss << nodeid << " ";
  //  }
  //  LOG(INFO) << oss.str();
  //}
  for (size_t i = 0; i < node_levels_.size(); ++i) {
    LOG(INFO) << "Level Node: [";
    for (uint32_t nodeid : node_levels_[i]) {
      const Node* node = graph[nodeid].source;
      LOG(INFO) << "\t#" << nodeid << ": \"" << node->attrs.name << "\""
                << (node->is_variable()? "(variable)" : "");
    }
    LOG(INFO) << "]";
    if (i < entry_group_levels_.size()) {
      LOG(INFO) << "Level NodeEntry: [";
      for (const uint32_t groupid : entry_group_levels_[i]) {
        ostringstream oss;
        oss << "\t{";
        for (const uint32_t entid : (*entry_groups_)[groupid]) {
          oss << "#" << entid << ": " << shapes[entid];
          //for (const uint32_t nid : entry_to_nodes_[entid]) {
            //oss << "n#" << nid << "|";
          //}
          oss << ", ";
        }
        LOG(INFO) << oss.str() << "},";
      }
      LOG(INFO) << "]";
    }
  }
  LOG(INFO) << "#Levels: " << node_levels_.size();
}*/

namespace {
inline bool AddInEntryIfNotExist(NodePtr node, const NodeEntry& toadd) {
  bool found = false;
  for (const auto& ent : node->inputs) {
    if (ent.node == toadd.node
        && ent.index == toadd.index
        && ent.version == toadd.version) {
      found = true;
      break;
    }
  }
  if (!found) {
    node->inputs.push_back(toadd);
    return true;
  }
  return false;
}
inline void FindEntry(
    const Node* n1, const Node* n2, unordered_set<NodeEntry>* ret) {
  for (const auto& ent : n2->inputs) {
    if (ent.node.get() == n1) {
      ret->insert(ent);
    }
  }
}
inline void FindEntry(
    const Node* n1, const vector<const Node*>& n2_set, unordered_set<NodeEntry>* ret) {
  for (const Node* n2 : n2_set) {
    FindEntry(n1, n2, ret);
  }
}
inline bool IsElementWiseNode(const Node* n) {
  if (n->is_variable()) return false;
  const auto& name = n->op()->name;
  return name == "Activation"
    || name == "elemwise_add"
    || name == "_mul"
    || name == "_backward_Activation"
    || name == "_backward_add"
    || name == "_backward_mul"
    || name == "ElementWiseSum"
    || name == "_grad_add";
}
}  // namespace

MegaGraph::MegaGraph(Graph* orig): orig_graph_(orig) {
  CHECK(orig->attrs.count("fwdent2bwdview") != 0);
  CHECK(orig->attrs.count("fwdnode2bwdview") != 0);
  CHECK(orig->attrs.count("fwdoutputs") != 0);
  CHECK(orig->attrs.count("bwdoutputs") != 0);

  typedef std::pair<std::vector<IndexedGraph::NodeEntry>,
                    std::vector<IndexedGraph::NodeEntry>> View;
  typedef std::unordered_map<IndexedGraph::NodeEntry, View> Fwdent2bwdview;
  typedef std::unordered_map<const Node*, View> Fwdnode2bwdview;
  const auto& fwdent2bwdview = orig->GetAttr<Fwdent2bwdview>("fwdent2bwdview");
  const auto& fwdnode2bwdview = orig->GetAttr<Fwdnode2bwdview>("fwdnode2bwdview");
  const auto& fwdoutputs = orig->GetAttr<vector<IndexedGraph::NodeEntry>>("fwdoutputs");

  const auto& origidx = orig->indexed_graph();
  unordered_map<const Node*, NodePtr> orignode2newnode;
  //unordered_map<NodeEntry, NodeEntry> origent2newent;
  //unordered_map<const Node*, uint32_t> newnode_num_succs;
  // Create mega graph.
  for (uint32_t orignid = 0; orignid < origidx.num_nodes(); ++orignid) {
    const Node* orignode = origidx[orignid].source;
    if (fwdnode2bwdview.count(orignode)) {
      NodePtr newnode = Node::Create();
      LOG(INFO) << "Orig node: " << orignode->attrs.name << " New node: " << newnode.get();
      orignode2newnode[orignode] = newnode;
      CHECK(!node_mappings_.count(newnode.get()));
      node_mappings_.insert(std::make_pair(newnode.get(), GraphView(orig, orignode)));
      const auto& bwdview = fwdnode2bwdview.at(orignode);
      node_mappings_.at(newnode.get()).Merge(
          GraphView(orig, bwdview.first, bwdview.second));
      LOG(INFO) << "\t" << node_mappings_.at(newnode.get());
      //ostringstream oss;
      //node_mappings_.at(newnode.get()).DFSNodeVisit([&] (const Node* n) {
            //oss << n->attrs.name << " ";
          //});
      //LOG(INFO) << "\t=> [" << oss.str() << "]";
      num_outputs_[newnode.get()] = orignode->num_outputs();
      //newnode_num_succs[newnode.get()] = 0;
      for (uint32_t i = 0; i < orignode->num_outputs(); ++i) {
        const auto& origent = IndexedGraph::NodeEntry{orignid, i, 0};
        NodeEntry newent{newnode, i, 0};
        entry_mappings_[newnode.get()].push_back(
            GraphView(orig, origent));
        if (fwdent2bwdview.count(origent)) {
          const auto& bwdview = fwdent2bwdview.at(origent);
          entry_mappings_.at(newnode.get()).at(i).Merge(
              GraphView(orig, bwdview.first, bwdview.second));
        }
        LOG(INFO) << "\tNew outentry#" << i;
        LOG(INFO) << "\t\t" << entry_mappings_.at(newnode.get()).at(i);
      }
      for (uint32_t i = 0; i < orignode->inputs.size(); ++i) {
        // Connect the entry.
        const auto& ent = orignode->inputs[i];
        const Node* in = ent.node.get();
        NodePtr newin = orignode2newnode.at(in);
        newnode->inputs.push_back(NodeEntry{newin, ent.index, 0});
      }
    }
  }
  // Graph output entries.
  for (const auto& origent : fwdoutputs) {
    const Node* orignode = origidx[origent.node_id].source;
    NodePtr newnode = orignode2newnode.at(orignode);
    graph_.outputs.push_back(
        NodeEntry{newnode, origent.index, 0});
  }

  const auto& newidx = graph_.indexed_graph();
  orignode_mapping_type_.resize(origidx.num_nodes(), MapType::kNA);
  orignode_mappings_.resize(origidx.num_nodes());
  origentry_mapping_type_.resize(origidx.num_node_entries(), MapType::kNA);
  origentry_mappings_.resize(origidx.num_node_entries());
  for (const auto& kv : node_mappings_) {
    const Node* newnode = kv.first;
    const auto& gv = kv.second;
    uint32_t newnid = newidx.node_id(newnode);
    gv.DFSNodeVisit([&] (const Node* orignode) {
          uint32_t nid = origidx.node_id(orignode);
          CHECK(orignode_mapping_type_[nid] == MapType::kNA)
            << origidx[nid].source->attrs.name << " " << gv;
          orignode_mapping_type_[nid] = MapType::kMapToNode;
          orignode_mappings_[nid] = newnode;
        });
    gv.DFSEntryVisit([&] (const Node* orignode, const vector<uint32_t>& oidx) {
          uint32_t nid = origidx.node_id(orignode);
          for (uint32_t i : oidx) {
            uint32_t eid = origidx.entry_id(nid, i);
            origentry_mapping_type_[eid] = MapType::kMapToNode;
            origentry_mappings_[eid] = newnode;
          }
        }, GraphView::VisitOption::kNoStartNoEnd);
    for (uint32_t idx = 0; idx < entry_mappings_.at(newnode).size(); ++idx) {
      const auto& gv = entry_mappings_.at(newnode)[idx];
      IndexedGraph::NodeEntry newent{newnid, idx, 0};
      gv.DFSNodeVisit([&] (const Node* orignode) {
            uint32_t nid = origidx.node_id(orignode);
            CHECK(orignode_mapping_type_[nid] == MapType::kNA)
              << origidx[nid].source->attrs.name << " " << gv;
            orignode_mapping_type_[nid] = MapType::kMapToEntry;
            orignode_mappings_[nid] = newent;
          });

      gv.DFSEntryVisit([&] (const Node* orignode, const vector<uint32_t>& oidx) {
            uint32_t nid = origidx.node_id(orignode);
            for (uint32_t i : oidx) {
              uint32_t eid = origidx.entry_id(nid, i);
              origentry_mapping_type_[eid] = MapType::kMapToEntry;
              origentry_mappings_[eid] = newent;
            }
          }, GraphView::VisitOption::kNoStart);

    }
  }
}

void MegaGraph::Print() {
  const auto& idx = graph_.indexed_graph();
  const auto& origidx = orig_graph_->indexed_graph();
  for (uint32_t nid = 0; nid < origidx.num_nodes(); ++nid) {
    auto* node = origidx[nid].source;
    switch (orignode_mapping_type_[nid]) {
      case MapType::kNA:
        LOG(INFO) << "Orig node#" << nid << " " << node->attrs.name << " NA"; break;
      case MapType::kMapToNode:
        LOG(INFO) << "Orig node#" << nid << " " << node->attrs.name << " ToNode"; break;
      case MapType::kMapToEntry:
        LOG(INFO) << "Orig node#" << nid << " " << node->attrs.name << " ToEntry"; break;
    }
    for (uint32_t i = 0; i < node->num_outputs(); ++i) {
      uint32_t eid = origidx.entry_id(nid, i);
      switch (origentry_mapping_type_[eid]) {
        case MapType::kNA:
          LOG(INFO) << "\tOrig entry#" << eid << " " << node->attrs.name << "#" << i << " NA"; break;
        case MapType::kMapToNode:
          LOG(INFO) << "\tOrig entry#" << eid << " " << node->attrs.name << "#" << i << " ToNode"; break;
        case MapType::kMapToEntry:
          LOG(INFO) << "\tOrig entry#" << eid << " " << node->attrs.name << "#" << i << " ToEntry"; break;
      }
    }
  }
  //////////////////// DEBUG PRINT
  //for (uint32_t nid = 0; nid < origidx.num_nodes(); ++nid) {
  //  const Node* n = origidx[nid].source;
  //  for (uint32_t i = 0; i < n->num_outputs(); ++i) {
  //    LOG(INFO) << "Entry#" << origidx.entry_id(nid, i)
  //      << " = Node#" << nid << "#" << i << "(" << n->attrs.name << ")["
  //      << (n->is_variable()? "var" : n->attrs.op->name) << "]";
  //  }
  //}
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    ostringstream oss;
    node_mappings_.at(node).DFSNodeVisit([&] (const Node* n) {
          oss << n->attrs.name << " ";
        });
    LOG(INFO) << "Node#" << nid << " => [" << oss.str() << "]";
    for (uint32_t i = 0; i < num_outputs_.at(node); ++i) {
      ostringstream oss;
      entry_mappings_.at(node).at(i).DFSEntryVisit(
          [&] (const Node* n, vector<uint32_t> idx) {
            for (uint32_t i : idx) {
              oss << n->attrs.name << "#" << i << " ";
            }
          });
      LOG(INFO) << "\t#" << i << " => [" << oss.str() << "]";
      //LOG(INFO) << "\t#" << i << " => " << entry_mappings_.at(node).at(i);
    }
  }
  //cout << "digraph {" << endl;
  //for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
  //  const Node* node = idx[nid].source;
  //  for (const auto& ent : node->inputs) {
  //    const uint32_t inid = idx.node_id(ent.node.get());
  //    cout << "\tn" << inid << " -> n" << nid << ";" << endl;
  //  }
  //}
  //cout << "}" << endl;
}
  
bool MegaGraph::IsElementWise(uint32_t nid) {
  const auto& idx = graph_.indexed_graph();
  const Node* node = idx[nid].source;
  bool ret = true;
  node_mappings_.at(node).DFSNodeVisit(
      [&] (const Node* orignode) {
        if (!IsElementWiseNode(orignode)) {
          ret = false;
        }
      });
  return ret;
}

ostream& operator << (ostream& os, const vector<uint32_t>& v) {
  os << "[";
  for (auto x : v) {
    os << x << " ";
  }
  return os << "]";
}

void MegaGraph::MergeElementwise() {
  const auto& idx = graph_.indexed_graph();
  const auto& origidx = orig_graph_->indexed_graph();
  unordered_map<string, vector<uint32_t>> elemgroups;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    string elemgroup = "";
    node_mappings_.at(node).DFSNodeVisit(
        [&] (const Node* orignode) {
          if (orignode->attrs.dict.count("elemwise_group")) {
            elemgroup = orignode->attrs.dict.at("elemwise_group");
          }
        });
    if (elemgroup != "") {
      // Sanity check
      CHECK(IsElementWise(nid));
      node_mappings_.at(node).DFSNodeVisit(
          [&] (const Node* orignode) {
            elemgroups[elemgroup].push_back(
                origidx.node_id(orignode));
          });
    }
  }
  for (const auto& kv : elemgroups) {
    LOG(INFO) << kv.first << ": " << kv.second;
  }
  // Create entry equals.
  for (const auto& kv : elemgroups) {
    const uint32_t anchor = origidx.entry_id(kv.second[0], 0);
    for (const uint32_t orignid : kv.second) {
      const Node* orignode = origidx[orignid].source;
      for (const auto& ent : orignode->inputs) {
        entry_equals_.push_back(std::make_pair(anchor, origidx.entry_id(ent)));
      }
      for (uint32_t i = 0; i < orignode->num_outputs(); ++i) {
        entry_equals_.push_back(std::make_pair(anchor, origidx.entry_id(orignid, i)));
      }
    }
  }
  // Create node equals.
  for (const auto& kv : elemgroups) {
    if (kv.second.empty()) continue;
    for (size_t i = 1; i < kv.second.size(); ++i) {
      node_equals_.push_back({kv.second[0], kv.second[i]});
    }
  }
}

void MegaGraph::MergeWeightAndGradients() {
  const auto& idx = graph_.indexed_graph();
  const auto& origidx = orig_graph_->indexed_graph();
  const auto& bwdoutputs = orig_graph_->GetAttr<vector<IndexedGraph::NodeEntry>>("bwdoutputs");
  vector<vector<uint32_t>> weightgroups;
  vector<vector<uint32_t>> sumgradgroups;
  for (const auto& origent : bwdoutputs) {
    uint32_t origeid = origidx.entry_id(origent);
    CHECK(origentry_mapping_type_[origeid] == MapType::kMapToEntry);
    const auto& newent = nnvm::get<IndexedGraph::NodeEntry>(origentry_mappings_[origeid]);
    const Node* newnode = idx[newent.node_id].source;
    LOG(INFO) << origidx[origent.node_id].source->attrs.name << "#"
      << origent.index << " eid=" << origeid << " -> " << newent.node_id << "#" << newent.index;
    vector<uint32_t> group;
    vector<uint32_t> sggroup;
    entry_mappings_.at(newnode).at(newent.index).DFSEntryVisit(
        [&] (const Node* orignode, const vector<uint32_t>& oidx) {
          uint32_t nid = origidx.node_id(orignode);
          for (uint32_t i : oidx) {
            LOG(INFO) << "\t" << orignode->attrs.name << "#" << i;
            uint32_t eid = origidx.entry_id(nid, i);
            group.push_back(eid);
          }
          if (utils::StartsWith(orignode->attrs.name, "sum_grad")) {
            sggroup.push_back(nid);
          }
        });
    weightgroups.push_back(group);
    sumgradgroups.push_back(sggroup);
  }

  // Create entry equals.
  for (const auto& g : weightgroups) {
    if (g.empty()) continue;
    const uint32_t anchor = g[0];
    for (size_t i = 1; i < g.size(); ++i) {
      entry_equals_.push_back({anchor, g[i]});
    }
  }
  // Create node equals.
  for (const auto& g : sumgradgroups) {
    if (g.empty()) continue;
    for (size_t i = 1; i < g.size(); ++i) {
      node_equals_.push_back({g[0], g[i]});
    }
  }
}

void MegaGraph::MergeRNNSteps() {
  const auto& idx = graph_.indexed_graph();
  const auto& origidx = orig_graph_->indexed_graph();
  typedef std::pair<std::vector<IndexedGraph::NodeEntry>,
                    std::vector<IndexedGraph::NodeEntry>> View;
  typedef std::unordered_map<const Node*, View> Fwdnode2bwdview;
  const auto& fwdnode2bwdview = orig_graph_->GetAttr<Fwdnode2bwdview>("fwdnode2bwdview");
  // 1. Find all step scopes.
  unordered_map<string, vector<uint32_t>> steps;
  unordered_map<string, vector<uint32_t>> step_entries;
  for (uint32_t nid = 0; nid < origidx.num_nodes(); ++nid) {
    const Node* node = origidx[nid].source;
    if (node->attrs.dict.count("rnn_step") && fwdnode2bwdview.count(node)) {
      const auto& stepname = node->attrs.dict.at("rnn_step");
      steps[stepname].push_back(nid);
      for (uint32_t i = 0; i < node->num_outputs(); ++i) {
        LOG(INFO) << node->attrs.name << "#" << i << " eid=" << origidx.entry_id(nid, i);
        step_entries[stepname].push_back(origidx.entry_id(nid, i));
      }
    }
  }
  if (steps.size() == 0) return;
  // 2. Find bp nodes.
  const size_t num_steps = steps.size();
  size_t max_step_len = 0;
  size_t max_step_ent_len = 0;
  for (const auto& kv : steps) {
    if (kv.second.size() > max_step_len) {
      max_step_len = kv.second.size();
    }
    if (step_entries[kv.first].size() > max_step_ent_len) {
      max_step_ent_len = step_entries[kv.first].size();
    }
  }
  vector<vector<uint32_t>> groups;
  /////////////////////////////////////////// Find node groups /////////////////////////////
  for (size_t t = 0; t < max_step_len; ++t) {
    vector<vector<uint32_t>> step_ex;
    for (const auto& kv : steps) {
      if (t >= kv.second.size()) continue;
      vector<uint32_t> vec;
      uint32_t nid = kv.second[t];
      CHECK(orignode_mapping_type_[nid] == MapType::kMapToNode)
        << origidx[nid].source->attrs.name;
      LOG(INFO) << origidx[nid].source->attrs.name;
      const Node* newnode = nnvm::get<const Node*>(orignode_mappings_[nid]);
      node_mappings_.at(newnode).DFSNodeVisit(
          [&] (const Node* orignode) {
            vec.push_back(origidx.node_id(orignode));
          });
      step_ex.push_back(vec);
    }
    LOG(INFO) << "At t=" << t;
    if (step_ex.empty()) continue;
    // sanity check
    for (const auto& vec : step_ex) CHECK_EQ(vec.size(), step_ex[0].size());
    for (size_t tt = 0; tt < step_ex[0].size(); ++tt) {
      uint32_t anchor = step_ex[0][tt];
      vector<uint32_t> group;
      for (const auto& vec : step_ex) {
        uint32_t nid = vec[tt];
        group.push_back(nid);
      }
      groups.push_back(group);
    }
  }
  /////////////////////////////////////////// Find entry groups /////////////////////////////
  for (size_t t = 0; t < max_step_ent_len; ++t) {
    vector<vector<uint32_t>> step_ex;
    for (const auto& kv : step_entries) {
      if (t >= kv.second.size()) continue;
      uint32_t eid = kv.second[t];
      LOG(INFO) << "Eid#" << eid;
      if (origentry_mapping_type_[eid] == MapType::kMapToEntry) {
        vector<uint32_t> vec;
        const auto& newent = nnvm::get<IndexedGraph::NodeEntry>(origentry_mappings_[eid]);
        entry_mappings_.at(idx[newent.node_id].source).at(newent.index)
          .DFSNodeVisit([&] (const Node* orignode) {
                vec.push_back(origidx.node_id(orignode));
              });
        // XXX(minjie): This is a dangerous skip.
        if (!vec.empty()) step_ex.push_back(vec);
      }
    }
    LOG(INFO) << "At t=" << t << " " << step_ex.size();
    if (step_ex.empty()) continue;
    // sanity check
    for (const auto& vec : step_ex) CHECK_EQ(vec.size(), step_ex[0].size());
    for (size_t tt = 0; tt < step_ex[0].size(); ++tt) {
      uint32_t anchor = step_ex[0][tt];
      vector<uint32_t> group;
      for (const auto& vec : step_ex) {
        uint32_t nid = vec[tt];
        group.push_back(nid);
      }
      groups.push_back(group);
    }
  }
  // 3. Build equals
  for (const auto& g : groups) {
    uint32_t anchor = g[0];
    const Node* anode = origidx[anchor].source;
    for (uint32_t nid : g) {
      const Node* n = origidx[nid].source;
      CHECK((anode->is_variable() && n->is_variable()) || anode->op() == n->op())
        << anode->attrs.name << " v.s. " << n->attrs.name;
      node_equals_.push_back(std::make_pair(anchor, nid));
      CHECK(anode->num_outputs() == n->num_outputs());
      CHECK(anode->inputs.size() == n->inputs.size());
      for (uint32_t i = 0; i < n->inputs.size(); ++i) {
        uint32_t eid1 = origidx.entry_id(anode->inputs[i]);
        uint32_t eid2 = origidx.entry_id(n->inputs[i]);
        entry_equals_.push_back({eid1, eid2});
      }
      for (uint32_t i = 0; i < n->num_outputs(); ++i) {
        uint32_t eid1 = origidx.entry_id(anchor, i);
        uint32_t eid2 = origidx.entry_id(nid, i);
        entry_equals_.push_back({eid1, eid2});
      }
    }
  }

  //vector<vector<uint32_t>> step_with_bp;
  //size_t max_step_len = 0;
  //for (const auto& kv : steps) {
  //  unordered_set<uint32_t> ns;
  //  for (uint32_t nid : kv.second) {
  //    CHECK(orignode_mapping_type_[nid] == MapType::kMapToNode)
  //      << origidx[nid].source->attrs.name;
  //    const Node* newnode = nnvm::get<const Node*>(orignode_mappings_[nid]);
  //    node_mappings_.at(newnode).DFSNodeVisit(
  //        [&] (const Node* orignode) {
  //          ns.insert(origidx.node_id(orignode));
  //        });
  //  }
  //  for (uint32_t eid : step_entries[kv.first]) {
  //    if (origentry_mapping_type_[eid] == MapType::kMapToEntry) {
  //      const auto& newent = nnvm::get<IndexedGraph::NodeEntry>(origentry_mappings_[eid]);
  //      entry_mappings_.at(idx[newent.node_id].source).at(newent.index)
  //        .DFSNodeVisit([&] (const Node* orignode) {
  //              ns.insert(origidx.node_id(orignode));
  //            });
  //    }
  //  }
  //  vector<uint32_t> ns_vec(ns.begin(), ns.end());
  //  std::sort(ns_vec.begin(), ns_vec.end());
  //  LOG(INFO) << kv.first << " " << ns_vec.size();
  //  for (uint32_t nid : ns_vec) {
  //    LOG(INFO) << "\t" << origidx[nid].source->attrs.name;
  //  }
  //  if (ns_vec.size() > max_step_len) {
  //    max_step_len = ns_vec.size();
  //  }
  //  step_with_bp.push_back(std::move(ns_vec));
  //}
  //// 3. Build equals
  //for (size_t t = 0; t < max_step_len; ++t) {
  //  uint32_t anchor = origidx.num_nodes();
  //  for (size_t s = 0; s < num_steps; ++s) {
  //    // NOTE: not all the RNN steps are identical to each other. For example, the hidden
  //    // state of the last step will no longer be fed to another step. Here we only merge
  //    // nodes that are available to the steps.
  //    if (t < step_with_bp[s].size()) {
  //      anchor = step_with_bp[s][t];
  //      break;
  //    }
  //  }
  //  CHECK(anchor < origidx.num_nodes());
  //  const Node* anode = origidx[anchor].source;
  //  for (size_t s = 1; s < num_steps; ++s) {
  //    if (t >= step_with_bp[s].size()) {
  //      // NOTE: not all the RNN steps are identical to each other. For example, the hidden
  //      // state of the last step will no longer be fed to another step. Here we only merge
  //      // nodes that are available to the steps.
  //      continue;
  //    }
  //    uint32_t nid = step_with_bp[s][t];
  //    const Node* n = origidx[nid].source;
  //    CHECK((anode->is_variable() && n->is_variable()) || anode->op() == n->op())
  //      << anode->attrs.name << " v.s. " << n->attrs.name;
  //    node_equals_.push_back(std::make_pair(anchor, nid));
  //    CHECK(anode->num_outputs() == n->num_outputs());
  //    CHECK(anode->inputs.size() == n->inputs.size());
  //    for (uint32_t i = 0; i < n->inputs.size(); ++i) {
  //      uint32_t eid1 = origidx.entry_id(anode->inputs[i]);
  //      uint32_t eid2 = origidx.entry_id(n->inputs[i]);
  //      entry_equals_.push_back({eid1, eid2});
  //    }
  //    for (uint32_t i = 0; i < n->num_outputs(); ++i) {
  //      uint32_t eid1 = origidx.entry_id(anchor, i);
  //      uint32_t eid2 = origidx.entry_id(nid, i);
  //      entry_equals_.push_back({eid1, eid2});
  //    }
  //  }
  //}
}
  
vector<uint32_t> MegaGraph::GetMegaNodeGroup(uint32_t node_id) const {
  const auto& origidx = orig_graph_->indexed_graph();
  vector<uint32_t> ret;
  CHECK(orignode_mapping_type_[node_id] == MapType::kMapToNode);
  const Node* meganode = nnvm::get<const Node*>(orignode_mappings_[node_id]);
  node_mappings_.at(meganode).DFSNodeVisit(
      [&] (const Node* orignode) {
        ret.push_back(origidx.node_id(orignode));
      });
  return ret;
}

}  // namespace pass
}  // namespace nnvm
