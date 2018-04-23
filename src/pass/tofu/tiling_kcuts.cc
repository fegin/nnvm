#include <nnvm/graph_attr_types.h>

#include "./tiling_kcuts.h"
#include "./utils.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {
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
}  // namespace

CutAlgorithm::CutAlgorithm(Graph* src,
                           const Levels& levels,
                           const NodeEntryGroups& eg,
                           const NodeGroups& ng,
                           size_t num_devices,
                           bool use_equal_cuts):
  src_graph_(src), levels_(levels),
  entry_groups_(eg), node_groups_(ng), use_equal_cuts_(use_equal_cuts),
  num_cuts_(utils::GetNumCuts(num_devices)) {
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
  dp_operators_.resize(levels.NumNodeGroupLevels());
  for (size_t i = 0; i < levels.NumNodeGroupLevels(); ++i) {
    const auto& nodelevel = levels.GetNodeGroupLevel(i);
    for (size_t j = 0; j < nodelevel.size(); ++j) {
      DPOp dpop;
      // Node id.
      const uint32_t node_group_id = nodelevel[j];
      dpop.node_group_id = node_group_id;
      const uint32_t node_id = *(node_groups_[node_group_id].begin());
      // Input/Output entries.
      const Node* node = idxgraph[node_id].source;
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
      const uint32_t groupid = dp_op.node_group_id;
      ostringstream oss;
      oss << "\t{";
      for (const uint32_t nodeid : node_groups_[groupid]) {
        const Node* node = graph[nodeid].source;
        oss << "#" << nodeid << ": \"" << node->attrs.name << "\""
            << (node->is_variable()? "(variable)" : "");
      }
      oss << " [";
      for (size_t choseid : dp_op.chosen_aligned_requests) {
        oss << choseid << " ";
      }
      oss << "]";
      LOG(INFO) << oss.str();
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

void CutAlgorithm::Run() {
  if (use_equal_cuts_) {
    cost_t total_cost = KEqualCuts(num_cuts_);
    LOG(INFO) << "Total K-equal-cuts cost: " << total_cost;
  } else {
    cost_t total_cost = KCuts(num_cuts_);
    LOG(INFO) << "Total K-cuts cost: " << total_cost;
  }
}

}  // namespace pass
}  // namespace nnvm
