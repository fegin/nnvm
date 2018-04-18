/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief Interface for tiling search algorithm.
 */
#ifndef NNVM_PASS_TILING_H_
#define NNVM_PASS_TILING_H_

#include <nnvm/scheme.h>
#include <nnvm/graph.h>

#include "./search_graph.h"

namespace nnvm {
namespace pass {

class Tiling {
 public:
  virtual ~Tiling() = default;
  // Run the tiling algorithm.
  virtual void Run() = 0;
  // (optional) Print the tiling result.
  virtual void Print() const { }
  // Get schemes of a node entry.
  virtual const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const = 0;
  // Get scheme requests of the given node.
  virtual const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const = 0;
  // Get scheme requests chosen for the given node.
  virtual const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const = 0;

  static std::unique_ptr<Tiling> Create(
      const std::string& name,
      Graph* graph,
      const Levels& levels,
      const NodeEntryGroups& groups,
      size_t num_devices);
};

class MergeTiling : public Tiling {
 public:
  MergeTiling(Graph* src, Tiling* t1, Tiling* t2): t1_(t1), t2_(t2) {
    const IndexedGraph& idxgraph = src->indexed_graph();
    entry_schemes_.resize(idxgraph.num_node_entries());
    scheme_requests_.resize(idxgraph.num_nodes());
    chosen_scheme_requests_.resize(idxgraph.num_nodes());
    for (uint32_t entry_id = 0; entry_id < idxgraph.num_node_entries(); ++entry_id) {
      entry_schemes_[entry_id] = t1->GetEntrySchemes(entry_id);
      const auto& from_t2 = t2->GetEntrySchemes(entry_id);
      entry_schemes_[entry_id].insert(
          entry_schemes_[entry_id].end(), from_t2.begin(), from_t2.end());
    }
    for (uint32_t node_id = 0; node_id < idxgraph.num_nodes(); ++node_id) {
      {
      scheme_requests_[node_id] = t1->GetSchemeRequests(node_id);
      const auto& from_t2 = t2->GetSchemeRequests(node_id);
      scheme_requests_[node_id].insert(
          scheme_requests_[node_id].end(), from_t2.begin(), from_t2.end());
      }
      {
      chosen_scheme_requests_[node_id] = t1->GetChosenSchemeRequests(node_id);
      const auto& from_t2 = t2->GetChosenSchemeRequests(node_id);
      chosen_scheme_requests_[node_id].insert(
          chosen_scheme_requests_[node_id].end(), from_t2.begin(), from_t2.end());
      }
    }
  }
  const std::vector<Scheme>& GetEntrySchemes(uint32_t entry_id) const {
    return entry_schemes_.at(entry_id);
  }
  const std::vector<SchemeRequest>& GetSchemeRequests(uint32_t node_id) const {
    return scheme_requests_.at(node_id);
  }
  const std::vector<size_t>& GetChosenSchemeRequests(uint32_t node_id) const {
    return chosen_scheme_requests_.at(node_id);
  }
 private:
  Tiling *t1_, *t2_;

  std::vector<std::vector<Scheme>> entry_schemes_;
  std::vector<std::vector<SchemeRequest>> scheme_requests_;
  std::vector<std::vector<size_t>> chosen_scheme_requests_;
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TILING_H_
