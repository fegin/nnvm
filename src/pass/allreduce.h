/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file allreduce.h
 * \brief Allreduce execution plans.
 */
#ifndef NNVM_PASS_ALLREDUCE_H_
#define NNVM_PASS_ALLREDUCE_H_

#include <nnvm/base.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace nnvm {
namespace pass {

class CommPlanner {
 public:
  typedef std::unordered_map<int, std::unordered_set<int>> Connectivity;
  const static std::string kDefaultPlanner;
  struct Reduce {
    std::vector<int> from;
    int to;
    Reduce(const std::vector<int>& from, int to): from(from), to(to) {}
  };
  struct ReduceStage {
    std::vector<Reduce> reduces;
  };
  struct Broadcast {
    int from;
    std::vector<int> to;
    Broadcast(int from, const std::vector<int>& to): from(from), to(to) {}
  };
  struct BroadcastStage {
    std::vector<Broadcast> broadcasts;
  };
  // Params:
  //   source:
  //     The id of processes that have the data to be reduced.
  //   target:
  //     The id of processes that have the reduced data should be stored.
  //
  // Returns:
  //   The plan of how to do allreduction. The result is a multi-stage reduction.
  //   The number of vectors equals to the number of stages. Each vector contains
  //   the id of processes that are in charge of reduction in that stage.
  //
  // For example (given source=[0, 1, 2, 3], target=1):
  //   Naive reduction returns:
  //     Stage #0: [{from: [0, 1, 2, 3], to=1}]
  //   Tree aggregation returns:
  //     Stage #0: [{from: [0, 1], to=1}, {source: [2, 3], to=2}]
  //     Stage #1: [{from: [1, 2], to=1}]
  virtual std::vector<ReduceStage> ReducePlan(
      const std::vector<int>& source, int target) = 0;

  // For example (given source=1, target=[0, 1, 2, 3]):
  //   Naive broadcast returns:
  //     Stage #0: [{from: 1, to: [0, 1, 2, 3]}]
  //   Tree broadcast returns:
  //     Stage #0: [{from: 1, to: [0, 2]}]
  //     Stage #1: [{from: 2, to: [3]}]
  virtual std::vector<BroadcastStage> BroadcastPlan(
      const std::vector<int>& source, const std::vector<int>& target) = 0;

  virtual std::vector<int> CommPlan(int source, int target) = 0;

  // If the given connectivity is empty, then all processes are connected.
  static std::unique_ptr<CommPlanner> CreatePlanner(
      const std::string& name,
      const Connectivity& connectivity = Connectivity());
};

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_ALLREDUCE_H_
