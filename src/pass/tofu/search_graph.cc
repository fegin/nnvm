#include "./search_graph.h"

#include <nnvm/graph_attr_types.h>
#include <queue>

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
    uint32_t num_node_entries,
    const std::vector<std::pair<uint32_t, uint32_t>>& equals)
  : groups_(num_node_entries), entry2group_(num_node_entries) {
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
  /*for (uint32_t eid = 0; eid < num_node_entries; ++eid) {
    cout << "Entry#" << eid << ": group#" << entry2group_[eid] << endl;
  }
  for (uint32_t gid = 0; gid < num_node_entries; ++gid) {
    cout << "Group#" << gid << " {";
    for (const auto& x : groups_[gid]) {
      cout << x << " ";
    }
    cout << "}" << endl;
  }*/
}

void Levels::AddNode(uint32_t levelid, uint32_t nodeid) {
  CHECK(node2index_.find(nodeid) == node2index_.end())
    << "Node #" << nodeid << " has already been added to level #" << levelid;
  if (levelid >= node_levels_.size()) {
    // New level.
    node_levels_.push_back(vector<uint32_t>());
  }
  CHECK_LT(levelid, node_levels_.size());
  const uint32_t level_index = node_levels_[levelid].size();
  node_levels_[levelid].push_back(nodeid);
  node2index_[nodeid] = make_pair(levelid, level_index);
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
  // For all entry in the group, make its index.
  for (const uint32_t ent : (*entry_groups_)[entry_group_id]) {
    CHECK(entry2index_.find(ent) == entry2index_.end())
      << "Entry should not be added twice (" << ent << ").";
    entry2index_[ent] = make_pair(levelid, level_index);
  }
}

void Levels::RemoveExtraNodeLevel() {
  if (node_levels_.size() > entry_group_levels_.size()) {
    // If the last level is a node level, it can be merged
    // with the second last one.
    const size_t old_size = node_levels_.size();
    const auto& last_lvl = node_levels_[old_size - 1];
    for (const uint32_t nid : last_lvl) {
      const size_t level_idx = node_levels_[old_size - 2].size();
      node_levels_[old_size - 2].push_back(nid);
      node2index_[nid] = std::make_pair(old_size - 2, level_idx);
    }
    node_levels_.resize(old_size - 1);
  }
  CHECK_EQ(node_levels_.size(), entry_group_levels_.size());
}

BFS::BFS(Graph* src, const NodeEntryGroups* groups): Levels(groups), src_graph_(src) {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  entry_to_nodes_.resize(graph.num_node_entries());
  node_to_entries_.resize(graph.num_nodes());
  for (uint32_t node_id = 0; node_id < graph.num_nodes(); ++node_id) {
    const IndexedGraph::Node& node = graph[node_id];
    // For all input entries, put the node in the adj list.
    for (const IndexedGraph::NodeEntry& in_ent : node.inputs) {
      const uint32_t in_ent_id = graph.entry_id(in_ent);
      entry_to_nodes_[in_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(in_ent_id);
      // Also put all the entries in the same group in the map.
      for (const uint32_t peer : (*groups)[groups->group_id(in_ent_id)]) {
        entry_to_nodes_[peer].insert(node_id);
        node_to_entries_[node_id].insert(peer);
      }
    }
    // For all output entries, put the node in the adj list.
    for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
      const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
      entry_to_nodes_[out_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(out_ent_id);
      // Also put all the entries in the same group in the map.
      for (const uint32_t peer : (*groups)[groups->group_id(out_ent_id)]) {
        entry_to_nodes_[peer].insert(node_id);
        node_to_entries_[node_id].insert(peer);
      }
    }
  }
}

void BFS::Run(uint32_t start_node_id) {
  queue<pair<uint32_t, uint32_t>> queue;  // (level, id)
  queue.push(make_pair(0, start_node_id));
  unordered_set<uint32_t> visited_nodes, visited_entries;
  while (!queue.empty()) {
    uint32_t level = 0, id = 0;
    tie(level, id) = queue.front();
    queue.pop();

    if (level % 2 == 0) {
      if (visited_nodes.count(id) > 0) {
        continue;
      }
      // This is a Node.
      visited_nodes.insert(id);
      AddNode(level / 2, id);
      // Put all its input/output entries into queue.
      for (const uint32_t ent_id : node_to_entries_[id]) {
        if (visited_entries.count(ent_id) == 0) {
          queue.push(make_pair(level + 1, ent_id));
        }
      }
    } else {
      if (visited_entries.count(id) > 0) {
        continue;
      }
      // This is a NodeEntry.
      visited_entries.insert(id);
      AddNodeEntry(level / 2, id);
      // Put all its producers/consumers into queue.
      for (const uint32_t node_id : entry_to_nodes_[id]) {
        if (visited_nodes.count(node_id) == 0) {
          queue.push(make_pair(level + 1, node_id));
        }
      }
    }
  }
  RemoveExtraNodeLevel();
}

void BFS::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  LOG(INFO) << "NodeEntry To Node";
  for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
    ostringstream oss;
    oss << "Entry#" << entid << ": ";
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      oss << nodeid << " ";
    }
    LOG(INFO) << oss.str();
  }
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
          oss << "#" << entid << ": " << shapes[entid] << ", ";
        }
        LOG(INFO) << oss.str() << "},";
      }
      LOG(INFO) << "]";
    }
  }
  LOG(INFO) << "#Levels: " << node_levels_.size();
}

NeuralLevels::NeuralLevels(Graph* src, const NodeEntryGroups* groups):
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
  /*for (size_t i = 0; i < node_groups_.size(); ++i) {
    LOG(INFO) << "Group #" << i << ": {";
    for (uint32_t nodeid : node_groups_[i]) {
      LOG(INFO) << "\t#" << nodeid << ": " << graph[nodeid].source->attrs.name << ",";
    }
    LOG(INFO) << "}";
  }*/
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
  /*for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
      LOG(INFO) << "Node #" << nodeid
                << ": " << graph[nodeid].source->attrs.name
                << " " << levelid[nodeid];
  }*/
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
  /*LOG(INFO) << "NodeEntry To Node";
  for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
    ostringstream oss;
    oss << "Entry#" << entid << ": ";
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      oss << nodeid << " ";
    }
    LOG(INFO) << oss.str();
  }*/
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
}

}  // namespace pass
}  // namespace nnvm
