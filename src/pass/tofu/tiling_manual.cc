#include "./tiling_manual.h"

#include <nnvm/graph_attr_types.h>
#include <nnvm/graph.h>

#include "./geometry.h"
#include "./utils.h"

using namespace std;

namespace nnvm {
namespace pass {

ManualTiling::ManualTiling(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  src_graph_(src),
  entry_groups_(groups),
  num_devices_(num_devices),
  num_cuts_(utils::GetNumCuts(num_devices)) {
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
  
DataParallelism::DataParallelism(Graph* src, const NodeEntryGroups& groups, size_t num_devices):
  ManualTiling(src, groups, num_devices) {
  param_schemes_ = vector<Scheme>(num_cuts_, Scheme::Rep());
  other_schemes_ = vector<Scheme>(num_cuts_, Scheme::Cut(0));
  // TODO(minjie): bias, batch_norm, etc.
  const IndexedGraph& idxgraph = src_graph_->indexed_graph();
  entry_schemes_.resize(idxgraph.num_node_entries(), &other_schemes_);
  for (uint32_t nodeid = 0; nodeid < idxgraph.num_nodes(); ++nodeid) {
    const Node* node = idxgraph[nodeid].source;
    if (node->is_variable() && utils::EndsWith(node->attrs.name, "weight")) {
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
      if (utils::EndsWith(node->attrs.name, "weight")) {
        const uint32_t entid = idxgraph.entry_id(nodeid, 0);
        const uint32_t ent_gid = entry_groups_.group_id(entid);
        for (const uint32_t id : entry_groups_[ent_gid]) {
          LOG(INFO) << "Find parameter entry: #" << id;
          entry_schemes_[id] = &param_schemes_;
        }
      } else {
        // Other variables do not have scheme.
      }
    } else if (!utils::EndsWith(node->attrs.name, "backward")) {
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
      if (utils::EndsWith(node->attrs.name, "weight")) {
        vector<Scheme>* param_schemes = nullptr;
        if (utils::Exists(node->attrs.name, "&mp&")) {
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
    } else if (utils::Exists(node->attrs.name, "&mp&")) {
      if (!utils::EndsWith(node->attrs.name, "backward")) {
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

}  // namespace pass
}  // namespace nnvm
