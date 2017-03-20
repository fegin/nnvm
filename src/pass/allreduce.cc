#include "./allreduce.h"
#include <fstream>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

class SimpleCommPlanner : public CommPlanner {
 public:
  explicit SimpleCommPlanner(const Connectivity& connectivity):
    connectivity_(connectivity), comm_plan_logs_("comm.log") {
    if (connectivity_.empty()) {
      for (int dev = 0; dev < 16; ++dev) {
        int group = dev / 8;
        for (int i = 0; i < 8; ++i) {
          int other = group * 8 + i;
          if (other != dev) {
            connectivity_[dev].insert(other);
            connectivity_[other].insert(dev);
          }
        }
        // jump link
        if (group == 0) {
          connectivity_[dev].insert(dev + 8);
          connectivity_[dev + 8].insert(dev);
        }
      }
    }
  }

  vector<ReduceStage> ReducePlan(const vector<int>& source, int target) override {
    vector<ReduceStage> plan;
    // TODO(minjie): VERY HACKY CODE! only support 8+8 group case.
    if (connectivity_.empty()) {
      // All processes are connected, just do simple reduce.
      plan.resize(1);
      plan[0].reduces.emplace_back(source, target);
    } else {
      // Multi-stage reduce based on the connectivity.
      vector<int> group_src[2];
      const int tgt_group = target / 8;
      const int other_group = 1 - tgt_group;
      const int tgt_bridge_point = (target + 8) % 16;
      for (const int src : source) {
        group_src[src / 8].push_back(src);
      }
      plan.resize(1);
      if (!group_src[tgt_group].empty()) {
        plan[0].reduces.emplace_back(group_src[tgt_group], target);
      }
      if (!group_src[other_group].empty()) {
        plan[0].reduces.emplace_back(group_src[other_group], tgt_bridge_point);
        // Another phase.
        plan.resize(2);
        if (group_src[tgt_group].empty()) {
          // No summation on the target group.
          plan[1].reduces.emplace_back(vector<int>{tgt_bridge_point}, target);
        } else {
          plan[1].reduces.emplace_back(vector<int>{target, tgt_bridge_point}, target);
        }
      }
    }

    LogReduce(source, target, plan);

    return plan;
  }
  
  vector<BroadcastStage> BroadcastPlan(const vector<int>& source,
                                       const vector<int>& target) override {
    // TODO(minjie): VERY HACKY CODE! only support 8+8 group case.
    vector<BroadcastStage> plan;

    vector<int> fetch_from(target.size(), -1);
    vector<int> source2, target2; // new source & target for second stage.
    // First: try local fetch.
    for (size_t i = 0; i < target.size(); ++i) {
      for (size_t j = 0; j < source.size(); ++j) {
        if (target[i] == source[j]) {
          fetch_from[i] = source[j];
          break;
        }
      }
    }
    // Second: Fetch within group.
    {
      size_t offset = 0;
      for (size_t i = 0; i < target.size(); ++i) {
        if (fetch_from[i] != -1) {
          continue;
        }
        for (size_t j = 0; j < source.size(); ++j) {
          const size_t srcidx = (offset + j) % source.size();
          if (connectivity_.at(target[i]).count(source[srcidx]) != 0) {
            fetch_from[i] = source[srcidx];
            offset = srcidx + 1;
            source2.push_back(target[i]);
            break;
          }
        }
        if (fetch_from[i] == -1) {
          // None of its neighbors has data.
          target2.push_back(target[i]);
        }
      }
    }
    // Check whether a bridge is needed between groups.
    bool need_bridge = false;
    for (const int tgt2 : target2) {
      bool connected = false;
      for (const int src2 : source2) {
        if (connectivity_.at(src2).count(tgt2) != 0) {
          connected = true;
          break;
        }
      }
      if (!connected) {
        need_bridge = true;
        break;
      }
    }
    // Make plan.
    plan.resize(1);
    for (const int src : source) {
      vector<int> push_to;
      for (size_t i = 0; i < target.size(); ++i) {
        if (fetch_from[i] == src) {
          push_to.push_back(target[i]);
        }
      }
      if (need_bridge) {
        const int bridge_point = (src + 8) % 16;
        push_to.push_back(bridge_point);
        source2.push_back(bridge_point);
        need_bridge = false;  // only need one bridge.
      }
      if (!push_to.empty()) {
        plan[0].broadcasts.emplace_back(src, std::move(push_to));
      }
    }

    // Second phase: Fetch within group again.
    if (!target2.empty()) {
      vector<int> fetch_from2(target2.size(), -1);
      {
        size_t offset = 0;
        for (size_t i = 0; i < target2.size(); ++i) {
          for (size_t j = 0; j < source2.size(); ++j) {
            const size_t srcidx = (offset + j) % source2.size();
            if (connectivity_.at(target2[i]).count(source2[srcidx]) != 0) {
              fetch_from2[i] = source2[srcidx];
              offset = srcidx + 1;
              break;
            }
          }
          CHECK_NE(fetch_from2[i], -1)
            << "Cannot finish the broadcast in 2 rounds.";
        }
      }
      // Make plan.
      plan.resize(2);
      for (const int src : source2) {
        vector<int> push_to;
        for (size_t i = 0; i < target2.size(); ++i) {
          if (fetch_from2[i] == src) {
            push_to.push_back(target2[i]);
          }
        }
        if (!push_to.empty()) {
          plan[1].broadcasts.emplace_back(src, std::move(push_to));
        }
      }
    }

    LogBroadcast(source, target, plan);
    
    return plan;
  }

  vector<int> CommPlan(int source, int target) override {
    vector<int> plan;
    if (connectivity_.empty() || connectivity_.at(source).count(target) != 0) {
      // Direct transfer.
    } else {
      // 2-round routing.
      for (int round : connectivity_.at(source)) {
        if (connectivity_.at(round).count(target) != 0) {
          plan.push_back(round);
          break;
        }
      }
      CHECK(!plan.empty()) << "cannot find a 2-round routing from "
        << source << " to " << target;
    }
    plan.push_back(target);
    return plan;
  }

  void LogReduce(const vector<int>& source, int target,
      const vector<ReduceStage>& stages) {
    ostringstream oss;
    oss << "Reduce: [";
    for (const int src : source) {
      oss << src << " ";
    }
    oss << "] -> " << target;
    //LOG(INFO) << oss.str();
    oss << "\nPlan:\n";
    for (size_t i = 0; i < stages.size(); ++i) {
      oss << "\tStage #" << i << ": ";
      for (const auto& red : stages[i].reduces) {
        oss << "[";
        for (int f : red.from) {
          oss << f << " ";
        }
        oss << "] +> " << red.to << "; ";
      }
      oss << endl;
    }
    comm_plan_logs_ << oss.str();
    comm_plan_logs_.flush();
  }

  void LogBroadcast(const vector<int>& source, const vector<int>& target,
      const vector<BroadcastStage>& stages) {
    ostringstream oss;
    oss << "Broadcast: [";
    for (const int src : source) {
      oss << src << " ";
    }
    oss << "] -> [";
    for (const int tgt : target) {
      oss << tgt << " ";
    }
    oss << "]";
    //LOG(INFO) << oss.str();
    oss << "\nPlan:\n";
    for (size_t i = 0; i < stages.size(); ++i) {
      oss << "\tStage #" << i << ": ";
      for (const auto& bcast : stages[i].broadcasts) {
        oss << bcast.from << " -> [";
        for (int t : bcast.to) {
          oss << t << " ";
        }
        oss << "]; ";
      }
      oss << endl;
    }
    comm_plan_logs_ << oss.str();
    comm_plan_logs_.flush();
  }

 private:
  Connectivity connectivity_;
  ofstream comm_plan_logs_;
};

}  // namespace

const string CommPlanner::kDefaultPlanner = "default";

unique_ptr<CommPlanner> CommPlanner::CreatePlanner(
    const string& ,
    const Connectivity& connectivity) {
  // name is currently not used.
  return unique_ptr<CommPlanner>(new SimpleCommPlanner(connectivity));
}

}  // namespace pass
}  // namespace nnvm
