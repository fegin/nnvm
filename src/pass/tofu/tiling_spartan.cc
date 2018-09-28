#include "./tiling_spartan.h"

#include <nnvm/graph_attr_types.h>
#include <nnvm/graph.h>
#include <queue>

#include "./utils.h"

using namespace std;

namespace nnvm {
namespace pass {

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
    num_devices_(num_devices), num_cuts_(utils::GetNumCuts(num_devices)) {
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
      if (sch.type == Scheme::kRep) {
        // Replication will become row partition
        entry_schemes_[in_eid] = vector<Scheme>(num_cuts_, Scheme::Cut(0));
      } else {
        entry_schemes_[in_eid] = vector<Scheme>(num_cuts_, sch);
      }
    }
  }
  for (size_t j = 0; j < node->num_outputs(); ++j) {
    const uint32_t out_eid = idx.entry_id(nid, j);
    if (entry_schemes_[out_eid].empty()) {
      const Scheme& sch = final_align.output_schemes[j];
      if (sch.type == Scheme::kRed) {
        //entry_schemes_[out_eid] = vector<Scheme>(num_cuts_, Scheme::Rep());
        // Replication will become row partition
        entry_schemes_[out_eid] = vector<Scheme>(num_cuts_, Scheme::Cut(0));
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

}  // namespace pass
}  // namespace nnvm
