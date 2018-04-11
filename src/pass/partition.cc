/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */

#include "./partition.h"

#include <queue>
#include <dmlc/json.h>
#include <nnvm/symbolic.h>

using namespace std;

namespace {
ostream& operator << (ostream& os, const nnvm::pass::Scheme& sch) {
  using nnvm::pass::Scheme;
  switch (sch.type) {
  case Scheme::kCut: return os << "C" << sch.dim;
  case Scheme::kRep: return os << "Rp";
  case Scheme::kRed: return os << "Rd";
  default:
    LOG(FATAL) << "Unknown scheme type: " << sch.type;
  }
  return os;
}

ostream& operator << (ostream& os, const nnvm::pass::Region& region) {
  return os << "[" << region.offset()
            << " + " << region.shape()
            << " in: " << region.entry_shape() << "]";
}

nnvm::TShape operator + (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] += shp2[i];
  }
  return ret;
}

nnvm::TShape operator - (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] -= shp2[i];
  }
  return ret;
}

nnvm::TShape operator / (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    CHECK(shp2[i] != 0 && ret[i] % shp2[i] == 0);
    ret[i] /= shp2[i];
  }
  return ret;
}

nnvm::TShape max(const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::max(ret[i], shp2[i]);
  }
  return ret;
}

nnvm::TShape min(const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::min(ret[i], shp2[i]);
  }
  return ret;
}
}  // namespace

namespace nnvm {
namespace pass {
namespace {
inline bool EndsWith(const string& value, const string& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
inline bool StartsWith(const string& value, const string& starting) {
  if (starting.size() > value.size()) return false;
  return std::equal(starting.begin(), starting.end(), value.begin());
}
inline bool Exists(const string& value, const string& sub) {
  if (sub.size() > value.size()) return false;
  return value.find(sub) != std::string::npos;
}
int _GetNumCuts(int num_devices) {
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

// Change the given scheme to a new one. Return false if all the possible schemes have
// been iterated.
bool NextScheme(const TShape& shape, Scheme* scheme) {
  CHECK_NE(scheme->type, Scheme::kRed);
  if (scheme->type == Scheme::kRep) {
    return false;
  }
  CHECK_EQ(scheme->type, Scheme::kCut);
  CHECK_GE(scheme->dim, 0);
  if (scheme->dim + 1 < shape.ndim() && scheme->dim + 1 < 2) { 
    // TODO(minjie): currently only consider partition on first two dimensions.
    ++scheme->dim;
  } else {
    scheme->dim = -1;
    scheme->type = Scheme::kRep;
  }
  return true;
}

bool NextSchemeVec(const vector<DPEntry>& entries, vector<Scheme>* schvec) {
  for (size_t i = 0; i < entries.size(); ++i) {
    if (NextScheme(entries[i].region.shape(), &(*schvec)[i])) {
      return true;
    } else {
      (*schvec)[i] = Scheme::Cut(0);
    }
  }
  return false;
}

// Return the pointer list to the contents given the index.
// prev_level and next_level could be nullptr when the function is called for the operators
// in the first and last BFS levels.
// If `ignore_jump` is set to true, entries that are not in the prev or next levels will be
// ignored. Rather than fail with error, it will put a nullptr as a placeholder.
template<typename T>
vector<const T*> ExtractFromIndex(
    const vector<Levels::Index>& index,
    const vector<T>* prev_level,
    const vector<T>* next_level,
    size_t current_level,
    bool ignore_jump) {
  vector<const T*> ret;
  for (const Levels::Index& idx : index) {
    if (idx.first == current_level) {
      // Content is in next level.
      CHECK_NOTNULL(next_level);
      ret.push_back(&(next_level->at(idx.second)));
    } else if (idx.first == current_level - 1) {
      // Content is in prev level.
      CHECK_NOTNULL(prev_level);
      ret.push_back(&(prev_level->at(idx.second)));
    } else if (ignore_jump) {
      ret.push_back(nullptr);
    } else {
      LOG(FATAL) << "Invalid entry index (" << idx.first << ", " << idx.second
                 << ") for operator in level #" << current_level;
    }
  }
  return ret;
}


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


bool Region::CanSplit2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    return region_shape_[sch.dim] % 2 == 0;
  case Scheme::kRep:
    return true;
  case Scheme::kRed:
    return false;
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return false;
}

pair<Region, Region> Region::Split2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    {
    TShape shp = region_shape_;
    CHECK_LT(sch.dim, region_shape_.ndim());
    CHECK(shp[sch.dim] % 2 == 0) << "Dimension " << sch.dim << " of size "
      << shp[sch.dim] << " cannot be splitted into two.";
    shp[sch.dim] /= 2;
    TShape offset = region_offset_;
    offset[sch.dim] += shp[sch.dim];
    Region r1(entry_shape_, region_offset_, shp);
    Region r2(entry_shape_, offset, shp);
    return make_pair(r1, r2);
    }
  case Scheme::kRep:
    {
    return make_pair(*this, *this);
    }
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return pair<Region, Region>();
}
  
cost_t Region::IntersectArea(const Region& r1, const Region& r2) {
  const TShape& r1_end = r1.offset() + r1.shape();
  const TShape& r2_end = r2.offset() + r2.shape();
  const TShape& st = max(r1.offset(), r2.offset());
  const TShape& ed = min(r1_end, r2_end);
  cost_t cost = 1;
  for (size_t i = 0; i < st.ndim(); ++i) {
    if (ed[i] <= st[i]) {
      // No intersection.
      return 0;
    } else {
      cost *= ed[i] - st[i];
    }
  }
  return cost;
}

// Note that it is possible that r1 and r2 have different areas. Consider following
// matmult example:
//  - First cut: C x R = red -> R
//  - Second cut: R x r = R
cost_t Region::ConvertCost2(const Region& r1, const Scheme& sch1,
                            const Region& r2, const Scheme& sch2) {
  CHECK_NE(sch2.type, Scheme::kRed)
    << "Reduction scheme is intermediate and could not be used as conversion target";
  cost_t cost = 0;
  if (sch1.type == Scheme::kRed) {
    // Reduction scheme requires special calculation.
    // Note that if source scheme is reduction, the area of source region and target
    // region may be different.
    if (sch2.type == Scheme::kCut) {
      if (!r2.CanSplit2(sch2)) {
        // Cannot split given the scheme. Return a very large cost that is guaranteed to
        // be worse.
        cost = 100 * (r1.Area() + r2.Area());
      } else {
        cost = r1.Area();
      }
    } else if (sch2.type == Scheme::kRep) {
      cost = 2 * r1.Area();
    } else {
      LOG(FATAL) << "Invalid target scheme: " << sch2;
    }
  } else {
    if (sch1.type == Scheme::kRep) {
      // If the source scheme is replication, then all data could be fetched locally.
    } else if (!r1.CanSplit2(sch1) || !r2.CanSplit2(sch2)) {
      // Cannot split given the scheme. Return a very large cost that is guaranteed to
      // be worse.
      cost = 100 * (r1.Area() + r2.Area());
    } else {
      const pair<Region, Region>& r1split = r1.Split2(sch1);
      const pair<Region, Region>& r2split = r2.Split2(sch2);
      cost += Region::IntersectArea(r1split.first, r2split.second);
      cost += Region::IntersectArea(r1split.second, r2split.first);
    }
    if (sch2.type == Scheme::kRep) {
      // If target scheme is replication, extra cost is required to replicate the area
      // that does not overlap with the source one (i.e, r2 - r1).
      cost += r2.Area() - Region::IntersectArea(r1, r2);
    }
  }
  CHECK_GE(cost, 0);
  return cost;
}

ManualTiling::ManualTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  src_graph_(src),
  entry_groups_(groups),
  num_devices_(num_devices),
  num_cuts_(_GetNumCuts(num_devices)) {
}

void ManualTiling::ChooseSchemeRequests() {
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    src_graph_->GetAttr<ShapeVector>("shape");
  chosen_scheme_requests_.resize(idxgraph.num_nodes());
  aligned_scheme_requests_.resize(idxgraph.num_nodes());
  cost_t total_cost = 0;
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) {
      continue;
    }
    // Choose inputs/outputs schemes and shapes.
    vector<Scheme> in_schemes(node->inputs.size());
    vector<Scheme> out_schemes(node->num_outputs());
    vector<TShape> in_shapes(node->inputs.size());
    vector<TShape> out_shapes(node->num_outputs());
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const uint32_t in_ent_id = idxgraph.entry_id(node->inputs[i]);
      // TODO only pick the first scheme.
      in_schemes[i] = this->GetEntrySchemes(in_ent_id)[0];
      in_shapes[i] = shapes[in_ent_id];
    }
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_ent_id = idxgraph.entry_id(nodeid, i);
      // TODO only pick the first scheme.
      out_schemes[i] = this->GetEntrySchemes(out_ent_id)[0];
      out_shapes[i] = shapes[out_ent_id];
    }

    // Get aligned scheme request.
    CHECK_NOTNULL(node->op());
    FAlignedSchemes align_func = align_map[node->op()];
    aligned_scheme_requests_[nodeid] = align_func(node->attrs, in_shapes, out_shapes);

    // Choose best aligned scheme.
    cost_t best_cost = std::numeric_limits<cost_t>::max();
    size_t chosen = 0;
    for (size_t i = 0; i < aligned_scheme_requests_[nodeid].size(); ++i) {
      cost_t cost = 0;
      const auto& align = aligned_scheme_requests_[nodeid][i];
      // Input conversion.
      for (size_t j = 0; j < node->inputs.size(); ++j) {
        Region reg(in_shapes[j]);
        cost += Region::ConvertCost2(reg,
                                     in_schemes[j],
                                     reg,
                                     align.input_schemes[j]);
        //LOG(INFO) << "\t(in coversion) cost=" << cost;
      }
      // Output conversion.
      for (size_t j = 0; j < node->num_outputs(); ++j) {
        Region reg(out_shapes[j]);
        cost += Region::ConvertCost2(reg,
                                     align.output_schemes[j],
                                     reg,
                                     out_schemes[j]);
        //LOG(INFO) << "\t(ou coversion) cost=" << cost;
      }
      if (cost < best_cost) {
        best_cost = cost;
        chosen = i;
      }
    }
    LOG(INFO) << "Node #" << nodeid << " " << node->attrs.name <<
      " choose " << chosen << " cost=" << best_cost;
    chosen_scheme_requests_[nodeid] = vector<size_t>(num_cuts_, chosen);
    total_cost += best_cost;
  }
  LOG(INFO) << "Estimated communication cost (2 nodes): " << total_cost;
}
  
const std::vector<Scheme>& MergeTiling::GetEntrySchemes(uint32_t entry_id) const {
  ostringstream oss;
  oss << "[";
  for (const auto& x : entry_schemes_.at(entry_id)) {
    oss << x << " ";
  }
  oss << "]";
  LOG(INFO) << "Entry#" << entry_id << ": " << oss.str();
  return entry_schemes_.at(entry_id);
}

DataParallelism::DataParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  ManualTiling(src, groups, num_devices) {
  param_schemes_ = vector<Scheme>(num_cuts_, Scheme::Rep());
  other_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(0));
  // TODO(minjie): bias, batch_norm, etc.
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  entry_schemes_.resize(idxgraph.num_node_entries(), &other_schemes_);
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable() && EndsWith(node->attrs.name, "weight")) {
      const uint32_t entid = idxgraph.entry_id(nodeid, 0);
      const uint32_t ent_gid = entry_groups_.group_id(entid);
      for (const uint32_t id : entry_groups_[ent_gid]) {
        LOG(INFO) << "Find parameter entry: #" << id;
        entry_schemes_[id] = &param_schemes_;
      }
    }
  }
  
  this->ChooseSchemeRequests();

  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) continue;
    if (node->op()->name == "FullyConnected"
        || node->op()->name == "Convolution") {
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 0);
    } else if (node->op()->name == "_backward_FullyConnected"
        || node->op()->name == "_backward_Convolution") {
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 2);
    }
    LOG(INFO) << "Overwrite Node #" << nodeid << " " << node->attrs.name
      << " to choose " << chosen_scheme_requests_[nodeid][0];
  }
}

const std::vector<Scheme>& DataParallelism::GetEntrySchemes(uint32_t entry_id) const {
  return *entry_schemes_[entry_id];
}

ModelParallelism::ModelParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  ManualTiling(src, groups, num_devices) {
  param_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(1));
  activation_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(1));
  other_schemes_ = vector<Scheme>(num_cuts_, Scheme::Rep());
  // TODO (minjie): bias, batch_norm, etc.
  // TODO (minjie): _plus, _sum_grad, etc.
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  entry_schemes_.resize(idxgraph.num_node_entries(), &other_schemes_);
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) {
      if (EndsWith(node->attrs.name, "weight")) {
        const uint32_t entid = idxgraph.entry_id(nodeid, 0);
        const uint32_t ent_gid = entry_groups_.group_id(entid);
        for (const uint32_t id : entry_groups_[ent_gid]) {
          LOG(INFO) << "Find parameter entry: #" << id;
          entry_schemes_[id] = &param_schemes_;
        }
      } else {
        // Other variables do not have scheme.
      }
    } else if (!EndsWith(node->attrs.name, "backward")) {
      CHECK_EQ(node->num_outputs(), 1);
      const uint32_t entid = idxgraph.entry_id(nodeid, 0);
      const uint32_t ent_gid = entry_groups_.group_id(entid);
      for (const uint32_t id : entry_groups_[ent_gid]) {
        LOG(INFO) << "Find activation entry: #" << id;
        entry_schemes_[id] = &activation_schemes_;
      }
    }
  }

  this->ChooseSchemeRequests();

  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) continue;
    if (node->op()->name == "FullyConnected"
        || node->op()->name == "Convolution") {
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 2);
    } else if (node->op()->name == "_backward_FullyConnected"
        || node->op()->name == "_backward_Convolution") {
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 1);
    }
    LOG(INFO) << "Overwrite Node #" << nodeid << " " << node->attrs.name
      << " to choose " << chosen_scheme_requests_[nodeid][0];
  }
}

const std::vector<Scheme>& ModelParallelism::GetEntrySchemes(uint32_t entry_id) const {
  return *entry_schemes_[entry_id];
}

struct JSONUserTilingNode {
  std::string name;
  std::vector<std::string> partitions;
  void Save(dmlc::JSONWriter *writer) const {
    LOG(FATAL) << "NNVM should only load a user tiling instead of saving.";
  }

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("name", &name);
    helper.DeclareField("partitions", &partitions);
    helper.ReadAllFields(reader);
  }
};

struct JSONUserTiling {
  std::vector<JSONUserTilingNode> nodes;
  void Save(dmlc::JSONWriter *writer) const {
    LOG(FATAL) << "NNVM should only load a user tiling instead of saving.";
  }

  void Load(dmlc::JSONReader *reader) {
    dmlc::JSONObjectReadHelper helper;
    helper.DeclareField("nodes", &nodes);
    helper.ReadAllFields(reader);
  }
};

UserTiling::UserTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  ManualTiling(src, groups, num_devices) {
  // Load the tiling scheme from the JSON file.
  CHECK_NE(src->attrs.count("user_tiling_json"), 0U)
      << "Load JSON require json to be presented.";
  const std::string &json_str = src->GetAttr<std::string>("user_tiling_json");
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JSONUserTiling jutiling;
  jutiling.Load(&reader);
  std::map<std::string, JSONUserTilingNode> tiling_map;
  // Partition all the node.
  for (auto node : jutiling.nodes) {
    tiling_map[node.name] = node;
  }
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  entry_schemes_.resize(idxgraph.num_node_entries());
  LOG(INFO) << "UserTiling ===========================================";
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    const JSONUserTilingNode junode = tiling_map[node->attrs.name];
    CHECK_EQ(node->num_outputs(), junode.partitions.size())
        << "Node" << nodeid << ", " << node->attrs.name <<
        ", partition size and outputs mismatched.";
    for (unsigned i = 0; i < node->num_outputs(); i++) {
      LOG(INFO) << "UserTiling is processing " << node->attrs.name;
      const uint32_t entid = idxgraph.entry_id(nodeid, i);
      const uint32_t ent_gid = entry_groups_.group_id(entid);
      const std::string partition = junode.partitions[i];
      CHECK_EQ(num_cuts_, partition.size());
      std::vector<Scheme> scheme;
      for (auto p : partition) {
        // TODO(fegin): Support more than two dimensions cut.
        switch (p) {
        case 'R':
          scheme.push_back(Scheme::Cut(0));
          break;
        case 'C':
          scheme.push_back(Scheme::Cut(1));
          break;
        case 'r':
          scheme.push_back(Scheme::Rep());
          break;
        default:
          LOG(FATAL) << "Not supported partition.";
        }
      }
      for (const uint32_t id : entry_groups_[ent_gid]) {
        LOG(INFO) << "Find parameter entry: #" << id;
        entry_schemes_[id] = scheme;
      }
    }
  }
  this->ChooseSchemeRequests();
}

HybridParallelism::HybridParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  ManualTiling(src, groups, num_devices) {
  dp_param_schemes_ = vector<Scheme>(num_cuts_, Scheme::Rep());
  dp_other_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(0));
  mp_param_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(1));
  mp_activation_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(1));
  mp_other_schemes_ = vector<Scheme>(num_cuts_, Scheme::Rep());
  // TODO (minjie): bias, batch_norm, etc.
  // TODO (minjie): _plus, _sum_grad, etc.
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  entry_schemes_.resize(idxgraph.num_node_entries(), &dp_other_schemes_);
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) {
      if (EndsWith(node->attrs.name, "weight")) {
        vector<Scheme>* param_schemes = nullptr;
        if (Exists(node->attrs.name, "&mp&")) {
          LOG(INFO) << "Find mp parameter node: " << node->attrs.name;
          param_schemes = &mp_param_schemes_;
        } else {
          LOG(INFO) << "Find dp parameter node: " << node->attrs.name;
          param_schemes = &dp_param_schemes_;
        }
        const uint32_t entid = idxgraph.entry_id(nodeid, 0);
        const uint32_t ent_gid = entry_groups_.group_id(entid);
        for (const uint32_t id : entry_groups_[ent_gid]) {
          entry_schemes_[id] = param_schemes;
        }
      } else {
        // Other variables do not have schemes.
      }
    } else if (Exists(node->attrs.name, "&mp&")) {
      if (!EndsWith(node->attrs.name, "backward")) {
        CHECK_EQ(node->num_outputs(), 1);
        const uint32_t entid = idxgraph.entry_id(nodeid, 0);
        const uint32_t ent_gid = entry_groups_.group_id(entid);
        for (const uint32_t id : entry_groups_[ent_gid]) {
          LOG(INFO) << "Find MP activation entry: #" << id << " " << node->attrs.name;
          entry_schemes_[id] = &mp_activation_schemes_;
        }
      } else {
        for (size_t i = 0; i < node->num_outputs(); ++i) {
          const uint32_t eid = idxgraph.entry_id(nodeid, i);
          for (const uint32_t id : entry_groups_[entry_groups_.group_id(eid)]) {
            LOG(INFO) << "Find other MP entry: #" << id << "" << node->attrs.name;
          entry_schemes_[id] = &mp_other_schemes_;
          }
        }
      }
    }
  }

  this->ChooseSchemeRequests();

  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) continue;
    if (node->op()->name == "FullyConnected") { // mp
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 2);
    } else if (node->op()->name == "Convolution") { // dp
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 0);
    } else if (node->op()->name == "_backward_FullyConnected") { // mp
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 1);
    } else if (node->op()->name == "_backward_Convolution") { // dp
      chosen_scheme_requests_[nodeid] = std::vector<size_t>(num_cuts_, 2);
    }
    LOG(INFO) << "Overwrite Node #" << nodeid << " " << node->attrs.name
      << " to choose " << chosen_scheme_requests_[nodeid][0];
  }
}

struct SpartanNode {
  uint32_t node_id;
  cost_t cost;
};

struct SpartanNodeCmp {
  bool operator() (const SpartanNode& sn1, const SpartanNode& sn2) const {
    return sn1.cost < sn2.cost;
  }
};


SpartanTiling::SpartanTiling(Graph* graph, const NodeEntryGroups& groups, size_t num_devices)
  : graph_(graph), groups_(groups),
    num_devices_(num_devices), num_cuts_(_GetNumCuts(num_devices)) {
  const auto& idx = graph->indexed_graph();
  entry_schemes_.resize(idx.num_node_entries());
  scheme_requests_.resize(idx.num_nodes());
  chosen_scheme_requests_.resize(idx.num_nodes());
  InitSchemeRequests();
}

  
void SpartanTiling::InitSchemeRequests() {
  const auto& idxgraph = graph_->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    graph_->GetAttr<ShapeVector>("shape");
  scheme_requests_.resize(idxgraph.num_nodes());
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable()) {
      continue;
    }
    vector<TShape> in_shapes(node->inputs.size());
    vector<TShape> out_shapes(node->num_outputs());
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const uint32_t in_ent_id = idxgraph.entry_id(node->inputs[i]);
      // TODO only pick the first scheme.
      in_shapes[i] = shapes[in_ent_id];
    }
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_ent_id = idxgraph.entry_id(nodeid, i);
      // TODO only pick the first scheme.
      out_shapes[i] = shapes[out_ent_id];
    }
    // Get aligned scheme request.
    CHECK_NOTNULL(node->op());
    FAlignedSchemes align_func = align_map[node->op()];
    scheme_requests_[nodeid] = align_func(node->attrs, in_shapes, out_shapes);
  }
}

void SpartanTiling::Run() {
  const ShapeVector& shapes = graph_->GetAttr<ShapeVector>("shape");
  const auto& idx = graph_->indexed_graph();
  typedef priority_queue<SpartanNode, vector<SpartanNode>, SpartanNodeCmp> SpartanQueue;
  SpartanQueue queue;
  vector<cost_t> node_costs(idx.num_nodes(), 0);
  vector<vector<const Node*>> entry2nodes(idx.num_node_entries());
  vector<bool> visited(idx.num_nodes(), false);
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const Node* node = idx[nid].source;
    if (node->is_variable()) {
      visited[nid] = true;
      continue;
    }
    cost_t cost = 0;
    for (const NodeEntry& in_ent : node->inputs) {
      const uint32_t in_eid = idx.entry_id(in_ent);
      const TShape& in_shape = shapes[in_eid];
      cost += in_shape.Size();
      entry2nodes[in_eid].push_back(node);
    }
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_eid = idx.entry_id(nid, i);
      const TShape& out_shape = shapes[out_eid];
      cost += out_shape.Size();
      entry2nodes[out_eid].push_back(node);
    }
    node_costs[nid] = cost;
    LOG(INFO) << "Node#" << nid << ": " << node->attrs.name << " cost=" << cost;
    queue.push(SpartanNode{nid, cost});
  }
  vector<bool> entry_visited(idx.num_node_entries(), false);
  cost_t total_cost = 0;
  while (!queue.empty()) {
    const SpartanNode& current = queue.top();
    queue.pop();
    const uint32_t nid = current.node_id;
    const Node* node = idx[nid].source;
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;
    total_cost += Decide(nid);
    // Adjust the priority of the adjacent nodes.
    unordered_set<uint32_t> adjusted;
    for (const NodeEntry& in_ent : node->inputs) {
      const uint32_t in_eid = idx.entry_id(in_ent);
      if (entry_visited[in_eid]) {
        continue;
      }
      const cost_t delta = shapes[in_eid].Size();
      for (uint32_t eid : groups_[groups_.group_id(in_eid)]) {
        entry_visited[eid] = true;
        for (const Node* tn : entry2nodes[eid]) {
          const uint32_t tnid = idx.node_id(tn);
          CHECK_GE(node_costs[tnid], delta);
          node_costs[tnid] -= delta;
          adjusted.insert(tnid);
        }
      }
    }
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_eid = idx.entry_id(nid, i);
      if (entry_visited[out_eid]) {
        continue;
      }
      const cost_t delta = shapes[out_eid].Size();
      for (uint32_t eid : groups_[groups_.group_id(out_eid)]) {
        entry_visited[eid] = true;
        for (const Node* tn : entry2nodes[eid]) {
          const uint32_t tnid = idx.node_id(tn);
          CHECK_GE(node_costs[tnid], delta);
          node_costs[tnid] -= delta;
          adjusted.insert(tnid);
        }
      }
    }
    // Push adjacent nodes back to the queue with new priority.
    for (uint32_t anid : adjusted) {
      if (!visited[anid]) {
        queue.push(SpartanNode{anid, node_costs[anid]});
      }
    }
  }
  LOG(INFO) << "Estimated communication cost (2 nodes): " << total_cost;
}

cost_t SpartanTiling::Decide(uint32_t nid) {
  const auto& idx = graph_->indexed_graph();
  const ShapeVector& shapes = graph_->GetAttr<ShapeVector>("shape");
  const Node* node = idx[nid].source;
  if (node->is_variable()) {
    return 0;
  }
  // Choose inputs/outputs schemes and shapes.
  vector<TShape> in_shapes(node->inputs.size());
  vector<TShape> out_shapes(node->num_outputs());
  for (size_t i = 0; i < node->inputs.size(); ++i) {
    const uint32_t in_ent_id = idx.entry_id(node->inputs[i]);
    in_shapes[i] = shapes[in_ent_id];
  }
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    const uint32_t out_ent_id = idx.entry_id(nid, i);
    out_shapes[i] = shapes[out_ent_id];
  }

  // Get aligned scheme request.
  const auto& sch_reqs = this->GetSchemeRequests(nid);

  // Choose best aligned scheme.
  cost_t best_cost = std::numeric_limits<cost_t>::max();
  vector<size_t> chosen;
  for (size_t i = 0; i < sch_reqs.size(); ++i) {
    cost_t cost = 0;
    const auto& align = sch_reqs[i];
    // Input conversion.
    for (size_t j = 0; j < node->inputs.size(); ++j) {
      const uint32_t in_entid = idx.entry_id(node->inputs[j]);
      const vector<Scheme>& in_sch = entry_schemes_[in_entid];
      if (in_sch.empty()) {
        // The entry has no scheme, so there is no conversion.
        continue;
      }
      // TODO: only pick the first scheme.
      Region reg(in_shapes[j]);
      cost += Region::ConvertCost2(reg,
                                   in_sch[0],
                                   reg,
                                   align.input_schemes[j]);
      //LOG(INFO) << "\t(in coversion) cost=" << cost;
    }
    // Output conversion.
    for (size_t j = 0; j < node->num_outputs(); ++j) {
      const uint32_t out_entid = idx.entry_id(nid, j);
      const vector<Scheme>& out_sch = entry_schemes_[out_entid];
      if (out_sch.empty()) {
        // The entry has no scheme, so there is no conversion.
        continue;
      }
      // TODO: only pick the first scheme.
      Region reg(out_shapes[j]);
      cost += Region::ConvertCost2(reg,
                                   align.output_schemes[j],
                                   reg,
                                   out_sch[0]);
      //LOG(INFO) << "\t(ou coversion) cost=" << cost;
    }
    if (cost < best_cost) {
      best_cost = cost;
      chosen = {i};
    } else if (cost == best_cost) {
      chosen.push_back(i);
    }
  }

  // Select from the best.
  size_t final_chosen = chosen[0];
  if (chosen.size() > 0) {
    // Make sure the chosen one replicates the smallest tensor.
    cost_t min_replicate_cost = std::numeric_limits<cost_t>::max();
    for (size_t chose : chosen) {
      cost_t replicate_cost = 0;
      const auto& align = sch_reqs[chose];
      for (size_t j = 0; j < node->inputs.size(); ++j) {
        const Scheme& sch = align.input_schemes[j];
        if (sch.type == Scheme::kRep) {
          replicate_cost += in_shapes[j].Size();
        }
      }
      for (size_t j = 0; j < node->num_outputs(); ++j) {
        const Scheme& sch = align.output_schemes[j];
        if (sch.type == Scheme::kRed) {
          replicate_cost += out_shapes[j].Size();
        }
      }
      if (replicate_cost < min_replicate_cost) {
        min_replicate_cost = replicate_cost;
        final_chosen = chose;
      }
    }
  }

  // Set the entry schemes to be the best one.
  const auto& final_align = sch_reqs[final_chosen];
  for (size_t j = 0; j < node->inputs.size(); ++j) {
    const uint32_t in_eid = idx.entry_id(node->inputs[j]);
    if (entry_schemes_[in_eid].empty()) {
      const Scheme& sch = final_align.input_schemes[j];
      entry_schemes_[in_eid] = vector<Scheme>(num_cuts_, sch);
    }
  }
  for (size_t j = 0; j < node->num_outputs(); ++j) {
    const uint32_t out_eid = idx.entry_id(nid, j);
    if (entry_schemes_[out_eid].empty()) {
      const Scheme& sch = final_align.output_schemes[j];
      if (sch.type == Scheme::kRed) {
        entry_schemes_[out_eid] = vector<Scheme>(num_cuts_, Scheme::Rep());
      } else {
        entry_schemes_[out_eid] = vector<Scheme>(num_cuts_, sch);
      }
    }
  }

  // Save the choice.
  LOG(INFO) << "Node #" << nid << " " << node->attrs.name <<
    " choose " << final_chosen << " cost=" << best_cost;
  chosen_scheme_requests_[nid] = vector<size_t>(num_cuts_, final_chosen);
  return best_cost;
}

CutAlgorithm::CutAlgorithm(Graph* src, const Levels& levels,
                           const NodeEntryGroups& groups):
  src_graph_(src), levels_(levels), entry_groups_(groups) {
  const IndexedGraph& idxgraph = src->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    src_graph_->GetAttr<ShapeVector>("shape");
  // Init DP structures.
  dp_entries_.resize(levels.NumEntryGroupLevels());
  for (size_t i = 0; i < levels.NumEntryGroupLevels(); ++i) {
    const auto& eglevel = levels.GetEntryGroupLevel(i);
    for (size_t j = 0; j < eglevel.size(); ++j) {
      DPEntry dpent;
      const uint32_t ent_group_id = eglevel[j];
      dpent.entry_group_id = ent_group_id;
      const uint32_t ent_id = *(entry_groups_[ent_group_id].begin());
      // The initial ghost region is the same as region.
      dpent.region = Region(shapes[ent_id]);
      dp_entries_[i].push_back(dpent);
    }
  }
  dp_operators_.resize(levels.NumNodeLevels());
  for (size_t i = 0; i < levels.NumNodeLevels(); ++i) {
    const auto& nodelevel = levels.GetNodeLevel(i);
    for (size_t j = 0; j < nodelevel.size(); ++j) {
      DPOp dpop;
      // Node id.
      const uint32_t node_id = nodelevel[j];
      dpop.node_id = node_id;
      // Input/Output entries.
      const Node* node = idxgraph[dpop.node_id].source;
      //LOG(INFO) << "!!" << node->attrs.name;
      vector<TShape> input_shapes, output_shapes;
      for (const NodeEntry& in_ent : node->inputs) {
        const uint32_t in_ent_id = idxgraph.entry_id(in_ent);
        const TShape& in_shape = shapes[in_ent_id];
        dpop.input_entry_index.push_back(levels.GetNodeEntryIndex(in_ent_id));
        // Initial ghost area shape is the same as node entry shape.
        dpop.input_ghost_regions.emplace_back(in_shape);
        input_shapes.push_back(in_shape);
        //LOG(INFO) << "!!in shape #" << in_ent_id << " " << in_shape;
      }
      for (size_t k = 0; k < node->num_outputs(); ++k) {
        const uint32_t out_ent_id = idxgraph.entry_id(node_id, k);
        const TShape& out_shape = shapes[out_ent_id];
        dpop.output_entry_index.push_back(levels.GetNodeEntryIndex(out_ent_id));
        // Initial ghost area shape is the same as node entry shape.
        dpop.output_ghost_regions.emplace_back(out_shape);
        output_shapes.push_back(out_shape);
        //LOG(INFO) << "!!out shape #" << out_ent_id << " " << out_shape;
      }
      // Aligned requests.
      if (node->is_variable()) {
        // Variable node. Any scheme should be aligned.
      } else {
        CHECK_NOTNULL(node->op());
        FAlignedSchemes align_func = align_map[node->op()];
        dpop.aligned_requests = align_func(node->attrs, input_shapes, output_shapes);
      }
      dp_operators_[i].push_back(dpop);
    }
  }
  CHECK_EQ(dp_operators_.size(), dp_entries_.size());
  // Init DP states.
  Init();
}

const std::vector<Scheme>& CutAlgorithm::GetEntrySchemes(uint32_t entry_id) const {
  const Levels::Index& index = levels_.GetNodeEntryIndex(entry_id);
  return dp_entries_[index.first][index.second].chosen_schemes;
}

const std::vector<SchemeRequest>& CutAlgorithm::GetSchemeRequests(
    uint32_t node_id) const {
  const Levels::Index& index = levels_.GetNodeIndex(node_id);
  return dp_operators_[index.first][index.second].aligned_requests;
}

// Get scheme requests chosen for the given node.
const std::vector<size_t>& CutAlgorithm::GetChosenSchemeRequests(
    uint32_t node_id) const {
  const Levels::Index& index = levels_.GetNodeIndex(node_id);
  return dp_operators_[index.first][index.second].chosen_aligned_requests;
}

void CutAlgorithm::Init() {
  dp_states_.resize(dp_entries_.size());
  for (size_t i = 0; i < dp_states_.size(); ++i) {
    vector<Scheme> schemes(dp_entries_[i].size(), Scheme::Cut(0));
    dp_states_[i].emplace_back(schemes);
    while (NextSchemeVec(dp_entries_[i], &schemes)) {
      // Create new state for each scheme combinations of the entries in this level.
      dp_states_[i].emplace_back(schemes);
    }
    LOG(INFO) << "DP Level #" << i << " size=" << dp_states_[i].size();
  }
}

void CutAlgorithm::Reset() {
  for (auto& lvl : dp_states_) {
    for (auto& state: lvl) {
      state.cost = 0;
      state.op_aligned_requests.clear();
    }
  }
}
  
cost_t CutAlgorithm::OneCut() {
  CHECK_GT(dp_states_.size(), 0);
  // Reset DP states.
  Reset();
  // Init state for BFS level 0.
  for (DPState& state: dp_states_[0]) {
    state.cost = 0;
    for (const DPOp& op : dp_operators_[0]) {
      if (IsVariable(op)) {
        // Variable operator. Any scheme should be fine, so no conversion cost.
        // Just put index 0 as the chosen aligned request.
        state.op_aligned_requests.push_back(0);
        continue;
      }
      cost_t op_cost = 0;
      size_t chosen_request = 0;
      tie(op_cost, chosen_request) = ConversionCost(op, nullptr, &state, 0);
      state.cost += op_cost;
      state.op_aligned_requests.push_back(chosen_request);
    }
  }
  // Do DP.
  //const size_t log_step = dp_states_.size() / 10;
  for (size_t i = 1; i < dp_states_.size(); ++i) {
    //if (i % log_step == 0) {
      LOG(INFO) << "DP Finished " << i;
    //}
    for (size_t j = 0; j < dp_states_[i].size(); ++j) {
      DPState& next_state = dp_states_[i][j];
      // Compute minimal cost to reach this state by looping all possible previous states.
      next_state.cost = std::numeric_limits<cost_t>::max();
      for (size_t k = 0; k < dp_states_[i-1].size(); ++k) {
        DPState& prev_state = dp_states_[i-1][k];
        cost_t state_cost = prev_state.cost;
        vector<size_t> op_requests;
        for (const DPOp& op : dp_operators_[i]) {
          if (IsVariable(op)) {
            // Variable operator. Any scheme should be fine, so conversion cost is zero.
            // Just put index 0 as the chosen aligned request.
            op_requests.push_back(0);
            continue;
          }
          //LOG(INFO) << src_graph_->indexed_graph()[op.node_id].source->attrs.name;
          cost_t op_cost = 0;
          size_t chosen_request = 0;
          tie(op_cost, chosen_request) = ConversionCost(op, &prev_state, &next_state, i);
          state_cost += op_cost;
          op_requests.push_back(chosen_request);
        }
        if (state_cost < next_state.cost) {
          // Record this.
          next_state.cost = state_cost;
          next_state.prev_state_index = k;
          next_state.op_aligned_requests = std::move(op_requests);
        }
      }
      //LOG(INFO) << "DP cost: level #" << i << " state #" << j << ": " << next_state.cost;
    }
  }
  // If the last level is node level, the total cost should also includes that.
  // TODO

  // Extract the optimal plan.
  return ExtractOptimalPlan();
}

pair<cost_t, size_t> CutAlgorithm::ConversionCost(
    const DPOp& op,
    const DPState* prev_state,
    const DPState* next_state,
    size_t lvl) const {
  //LOG(INFO) << src_graph_->indexed_graph()[op.node_id].source->attrs.name;
  const vector<Scheme>* prev_schemes = (prev_state)? &prev_state->schemes : nullptr;
  const vector<Scheme>* next_schemes = (next_state)? &next_state->schemes : nullptr;
  const vector<DPEntry>* prev_entries = (lvl > 0)? &dp_entries_[lvl-1] : nullptr;
  const vector<DPEntry>* next_entries = (lvl < dp_entries_.size())? &dp_entries_[lvl] : nullptr;
  const bool ignore_jump = levels_.AllowCrossLevelEdges();
  // Extract schemes for inputs and outputs of the op.
  const vector<const Scheme*>& input_schemes =
    ExtractFromIndex<Scheme>(op.input_entry_index, prev_schemes, next_schemes, lvl, ignore_jump);
  const vector<const Scheme*>& output_schemes =
    ExtractFromIndex<Scheme>(op.output_entry_index, prev_schemes, next_schemes, lvl, ignore_jump);
  // Extract entries for inputs and outputs of the op.
  const vector<const DPEntry*>& input_entries =
    ExtractFromIndex<DPEntry>(op.input_entry_index, prev_entries, next_entries, lvl, ignore_jump);
  const vector<const DPEntry*>& output_entries =
    ExtractFromIndex<DPEntry>(op.output_entry_index, prev_entries, next_entries, lvl, ignore_jump);
  const vector<SchemeRequest>& aligned_requests = op.aligned_requests;
  CHECK_EQ(input_schemes.size(), input_entries.size());
  CHECK_EQ(input_schemes.size(), op.input_ghost_regions.size());
  CHECK_EQ(output_schemes.size(), output_entries.size());
  CHECK_EQ(output_schemes.size(), op.output_ghost_regions.size());
  CHECK_GT(aligned_requests.size(), 0);
  cost_t cost = std::numeric_limits<cost_t>::max();
  size_t req_idx = 0;
  for (size_t i = 0; i < aligned_requests.size(); ++i) {
    const SchemeRequest& req = aligned_requests[i];
    CHECK_EQ(input_schemes.size(), req.input_schemes.size());
    CHECK_EQ(output_schemes.size(), req.output_schemes.size());
    cost_t req_cost = 0;
    // Input conversion cost.
    for (size_t j = 0; j < input_schemes.size(); ++j) {
      if (input_schemes[j] == nullptr) {
        // Cannot get the scheme from either prev or next state. This may
        // because of the shortcut (jump edge) in the graph. In this case, simply
        // ignore the cost.
        CHECK(ignore_jump);
        continue;
      }
      req_cost += Region::ConvertCost2(input_entries[j]->region,
                                       *input_schemes[j],
                                       op.input_ghost_regions[j],
                                       req.input_schemes[j]);
    }
    // Output conversion cost.
    for (size_t j = 0; j < output_schemes.size(); ++j) {
      if (output_schemes[j] == nullptr) {
        // Cannot get the scheme from either prev or next state. This may
        // because of the shortcut (jump edge) in the graph. In this case, simply
        // ignore the cost.
        CHECK(ignore_jump);
        continue;
      }
      req_cost += Region::ConvertCost2(op.output_ghost_regions[j],
                                       req.output_schemes[j],
                                       output_entries[j]->region,
                                       *output_schemes[j]);
    }
    // Save the minimal cost.
    if (req_cost < cost) {
      cost = req_cost;
      req_idx = i;
    }
  }
  return make_pair(cost, req_idx);
}

cost_t CutAlgorithm::ExtractOptimalPlan() {
  size_t num_levels = dp_states_.size();
  cost_t min_cost = std::numeric_limits<cost_t>::max();
  DPState* min_state = nullptr;
  for (DPState& state : dp_states_[num_levels-1]) {
    if (state.cost < min_cost) {
      min_cost = state.cost;
      min_state = &state;
    }
  }
  LOG(INFO) << "Min cost: " << min_cost;
  for (int i = dp_states_.size() - 1; i >= 0; --i) {
    CHECK_EQ(dp_entries_[i].size(), min_state->schemes.size());
    CHECK_EQ(dp_operators_[i].size(), min_state->op_aligned_requests.size());
    // Record best scheme for each entry.
    for (size_t j = 0; j < dp_entries_[i].size(); ++j) {
      dp_entries_[i][j].chosen_schemes.push_back(
          min_state->schemes[j]);
    }
    // Record best aligned request for each operator. Variable operator will be ignored.
    for (size_t j = 0; j < dp_operators_[i].size(); ++j) {
      if (!IsVariable(dp_operators_[i][j])) {
        dp_operators_[i][j].chosen_aligned_requests.push_back(
            min_state->op_aligned_requests[j]);
      }
    }
    if (i > 0) {
      min_state = &dp_states_[i-1][min_state->prev_state_index];
    }
  }
  // TODO handle situation where the last BFS level is an operator level.
  return min_cost;
}

void CutAlgorithm::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  for (size_t i = 0; i < dp_operators_.size(); ++i) {
    LOG(INFO) << "Level Node: [";
    for (const auto& dp_op : dp_operators_[i]) {
      const uint32_t nodeid = dp_op.node_id;
      const Node* node = graph[nodeid].source;
      ostringstream oss;
      oss << " [";
      for (size_t choseid : dp_op.chosen_aligned_requests) {
        oss << choseid << " ";
      }
      oss << "]";
      LOG(INFO) << "\t#" << nodeid << ": \"" << node->attrs.name << "\""
                << (node->is_variable()? "(variable)" : "")
                << oss.str();
    }
    LOG(INFO) << "]";
    if (i < dp_entries_.size()) {
      LOG(INFO) << "Level NodeEntry: [";
      for (const auto& dp_ent : dp_entries_[i]) {
        const uint32_t groupid = dp_ent.entry_group_id;
        ostringstream oss;
        oss << "\t{";
        for (const uint32_t entid : entry_groups_[groupid]) {
          oss << "#" << entid << " ";
        }
        oss << "}, " << dp_ent.region << "[";
        for (const Scheme& sch : dp_ent.chosen_schemes) {
          oss << sch << " ";
        }
        oss << "]";
        LOG(INFO) << oss.str();
      }
      LOG(INFO) << "]";
    }
  }
}

// K-cut algorithm.
cost_t CutAlgorithm::KCuts(uint32_t K) {
  if (K == 0) {
    return 0;
  }
  // Compute one-cut.
  cost_t cut_cost = OneCut();
  // Prune entry regions.
  for (auto& ent_lvl : dp_entries_) {
    for (DPEntry& ent : ent_lvl) {
      const Scheme& cur_sch = ent.chosen_schemes[ent.chosen_schemes.size() - 1];
      ent.region = ent.region.Split2(cur_sch).first;
    }
  }
  // Compute ghost regions.
  for (auto& op_lvl : dp_operators_) {
    for (DPOp& op : op_lvl) {
      if (IsVariable(op)) {
        // For variable operator, all schemes are aligned. In fact, the ghost area is not
        // being considered when computing the conversion cost (the cost is always zero).
        // Therefore, no need to compute ghost regions for this.
        continue;
      }
      CHECK(!op.chosen_aligned_requests.empty());
      size_t cur_req = op.chosen_aligned_requests[op.chosen_aligned_requests.size() - 1];
      CHECK_LT(cur_req, op.aligned_requests.size());
      const SchemeRequest& req = op.aligned_requests[cur_req];
      // Inputs.
      for (size_t i = 0; i < op.input_ghost_regions.size(); ++i) {
        op.input_ghost_regions[i] = op.input_ghost_regions[i].Split2(
            req.input_schemes[i]).first;
      }
      // Outputs.
      for (size_t i = 0; i < op.output_ghost_regions.size(); ++i) {
        const Scheme& out_sch = req.output_schemes[i];
        if (out_sch.type != Scheme::kRed) {
          // The ghost area of the Reduction scheme is unchanged. Otherwise, split
          // the ghost area to form the new one.
          op.output_ghost_regions[i] =
            op.output_ghost_regions[i].Split2(out_sch).first;
        }
      }
    }
  }
  // Compute (k-1)-cut.
  return cut_cost + 2 * KCuts(K - 1);
}

cost_t CutAlgorithm::KEqualCuts(uint32_t K) {
  if (K == 0) {
    return 0;
  }
  // Compute one-cut.
  cost_t cut_cost = OneCut();
  // Populate the one-cut result to all k-cuts.
  for (size_t i = 0; i < dp_states_.size(); ++i) {
    for (size_t j = 0; j < dp_entries_[i].size(); ++j) {
      for (uint32_t k = 0; k < K - 1; ++k) {
        dp_entries_[i][j].chosen_schemes.push_back(dp_entries_[i][j].chosen_schemes[0]);
      }
    }
    for (size_t j = 0; j < dp_operators_[i].size(); ++j) {
      if (!IsVariable(dp_operators_[i][j])) {
        for (uint32_t k = 0; k < K - 1; ++k) {
          dp_operators_[i][j].chosen_aligned_requests.push_back(
              dp_operators_[i][j].chosen_aligned_requests[0]);
        }
      }
    }
  }
  return cut_cost;
}

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
  std::unordered_set<Node*> root;
  for (size_t i = 0; i < from.TotalNumBlocks(); ++i) {
    root.insert(from.BlockAt(i).entry.node.get());
  }
  for (size_t i = 0; i < to->TotalNumBlocks(); ++i) {
    auto& toblk = to->BlockAt(i);
    std::vector<NodeEntry> blkroot;
    std::vector<Node*> seq;
    DFSVisitWithRoot({toblk.entry}, root,
        [&] (const NodePtr& node) {
          for (const auto& inent : node->inputs) {
            if (root.count(inent.node.get())) {
              blkroot.push_back(inent);
            }
          }
          seq.push_back(node.get());
        });
    //LOG(INFO) << "#Root blks: " << blkroot.size();
    //LOG(INFO) << "Fused: [";
    //for (Node* n : seq) {
      //LOG(INFO) << "\t" << n->attrs.op->name;
    //}
    //LOG(INFO) << "]";
    const Op* fused_convert_op = Op::Get("_TofuFusedConvert");
    // Fused op name.
    ostringstream oss;
    oss << "_TOFU_CONVERT";
    NodePtr node = Node::Create();
    // all input entries
    node->inputs = std::move(blkroot);
    node->attrs.op = fused_convert_op;
    node->attrs.name = oss.str();
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
  CHECK(!inputs.empty());
  // Sub-operators can simply fetch blocks under the same devices
  // since all required data should be fetched to locally already.
  const uint32_t num_devices = nodes.size();
  // TODO(minjie): NodeEntry version?
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
    LOG(INFO) << "Process node#" << nodeid << ": " << node->attrs.name;
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
    for (size_t i = 0; i < num_inputs; ++i) {
      const uint32_t in_ent_id = graph.entry_id(node->inputs[i]);
      const TShape& shape = shapes[in_ent_id];
      input_grids.emplace_back(shape, input_schemes[i]);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
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
      /*
      for (NodePtr depend_node : node->control_deps) {
        const uint32_t depend_nid = graph.node_id(depend_node.get());
        CHECK_LT(depend_nid, nodeid);
        n->control_deps.push_back(splitted_nodes[depend_nid][i]);
      }*/
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
