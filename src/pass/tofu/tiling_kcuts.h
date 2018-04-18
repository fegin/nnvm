/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief K-cuts tiling.
 */
#ifndef NNVM_PASS_TOFU_TILING_KCUTS_H_
#define NNVM_PASS_TOFU_TILING_KCUTS_H_

#include "./geometry.h"
#include "./tiling.h"

namespace nnvm {
namespace pass {

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
  std::vector<size_t> op_aligned_requests;

  explicit DPState(const std::vector<Scheme>& schemes): schemes(schemes) {}
};

class CutAlgorithm : public Tiling {
 public:
  // Constructor.
  CutAlgorithm(Graph* src,
               const Levels& levels,
               const NodeEntryGroups& groups,
               size_t num_devices,
               bool use_equal_cuts=false);

  // One cut algorithm. Return the minimal cost.
  cost_t OneCut();

  // K-cut algorithm. Return the minimal cost.
  cost_t KCuts(uint32_t K);

  // Apply the same one-cut result to all K cuts. Return the one-cut cost.
  cost_t KEqualCuts(uint32_t K);

  // Get schemes of a node entry.
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const override;
  
  // Get scheme requests of the given node.
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const override;

  // Get scheme requests chosen for the given node.
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const override;

  void Run() override;

  // Print debug information.
  void Print() const override;

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
  const bool use_equal_cuts_{false};
  const uint32_t num_cuts_;

  std::vector<std::vector<DPOp>> dp_operators_;
  std::vector<std::vector<DPEntry>> dp_entries_;

  std::vector<std::vector<DPState>> dp_states_;
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TOFU_TILING_KCUTS_H_
