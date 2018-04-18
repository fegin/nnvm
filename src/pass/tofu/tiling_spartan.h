/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief Manual tiling.
 */
#ifndef NNVM_PASS_TOFU_TILING_SPARTAN_H_
#define NNVM_PASS_TOFU_TILING_SPARTAN_H_

#include "./tiling.h"
#include "./geometry.h"

namespace nnvm {
namespace pass {

class SpartanTiling : public Tiling {
 public:
  SpartanTiling(Graph* graph, const NodeEntryGroups& groups, size_t num_devices);

  void Run() override;

  // Get schemes of a node entry.
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const override {
    return entry_schemes_[entry_id];
  }
  // Get scheme requests of the given node.
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const override {
    return scheme_requests_[node_id];
  }
  // Get scheme requests chosen for the given node.
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const override {
    return chosen_scheme_requests_[node_id];
  }

 private:
  void InitSchemeRequests();
  cost_t Decide(uint32_t nid);

  Graph* graph_;
  const NodeEntryGroups& groups_;
  const size_t num_devices_;
  const size_t num_cuts_;

  std::vector<std::vector<Scheme>> entry_schemes_;
  std::vector<std::vector<SchemeRequest>> scheme_requests_;
  std::vector<std::vector<size_t>> chosen_scheme_requests_;
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TOFU_TILING_SPARTAN_H_
