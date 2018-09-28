/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief Manual tiling.
 */
#ifndef NNVM_PASS_TOFU_TILING_MANUAL_H_
#define NNVM_PASS_TOFU_TILING_MANUAL_H_

#include "./tiling.h"

namespace nnvm {
namespace pass {

class ManualTiling : public Tiling {
 public:
  ManualTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<size_t> GetChosenSchemeRequests(uint32_t node_id) const override {
    return chosen_scheme_requests_.at(node_id);
  }
  std::vector<SchemeRequest> GetSchemeRequests(uint32_t node_id) const override {
    return aligned_scheme_requests_.at(node_id);
  }
 protected:
  void ChooseSchemeRequests();
  Graph* src_graph_;
  const NodeEntryGroups& entry_groups_;
  const size_t num_devices_;
  const size_t num_cuts_;

  std::vector<std::vector<SchemeRequest>> aligned_scheme_requests_;
  std::vector<std::vector<size_t>> chosen_scheme_requests_;
};

class DataParallelism : public ManualTiling {
 public:
  DataParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override;
  void Run() override {}
  
 private:
  std::vector<Scheme> param_schemes_;
  std::vector<Scheme> other_schemes_;
  std::vector<std::vector<Scheme>*> entry_schemes_;
};

class ModelParallelism : public ManualTiling {
 public:
  ModelParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override;
  void Run() override {}

 private:
  std::vector<Scheme> param_schemes_;
  std::vector<Scheme> activation_schemes_;
  std::vector<Scheme> other_schemes_;
  std::vector<std::vector<Scheme>*> entry_schemes_;
};

class HybridParallelism : public ManualTiling {
 public:
  HybridParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override {
    return *entry_schemes_[entry_id];
  }
  void Run() override {}

 private:
  std::vector<Scheme> dp_param_schemes_;
  std::vector<Scheme> dp_other_schemes_;
  std::vector<Scheme> mp_param_schemes_;
  std::vector<Scheme> mp_activation_schemes_;
  std::vector<Scheme> mp_other_schemes_;
  std::vector<std::vector<Scheme>*> entry_schemes_;
};

class UserTiling : public ManualTiling {
 public:
  UserTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override {
    return entry_schemes_[entry_id];
  }
  void Run() override {}

 private:
  std::vector<std::vector<Scheme>> entry_schemes_;
};

class AllRowTiling : public ManualTiling {
 public:
  AllRowTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override {
    return entry_schemes_;
  }
  void Run() override {}

 private:
  std::vector<Scheme> entry_schemes_;
};

class AllRowAllFirstTiling : public ManualTiling {
 public:
  AllRowAllFirstTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices);
  std::vector<Scheme> GetEntrySchemes(uint32_t entry_id) const override {
    return entry_schemes_;
  }
  void Run() override {}

 private:
  std::vector<Scheme> entry_schemes_;
};
}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TOFU_TILING_MANUAL_H_
