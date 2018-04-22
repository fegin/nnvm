#include <dmlc/logging.h>
#include <string>

#ifndef NNVM_PASS_TOFU_UTILS_H_
#define NNVM_PASS_TOFU_UTILS_H_

namespace nnvm {
namespace pass {
namespace utils {
inline bool EndsWith(const std::string& value, const std::string& ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool StartsWith(const std::string& value, const std::string& starting) {
  if (starting.size() > value.size()) return false;
  return std::equal(starting.begin(), starting.end(), value.begin());
}

inline bool Exists(const std::string& value, const std::string& sub) {
  if (sub.size() > value.size()) return false;
  return value.find(sub) != std::string::npos;
}

inline int GetNumCuts(int num_devices) {
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

inline std::vector<uint32_t> UnionFind(
    uint32_t n,
    const std::vector<std::pair<uint32_t, uint32_t>>& equals) {
  std::vector<uint32_t> groups(n, 0);
  for (uint32_t i = 0; i < groups.size(); ++i) {
    groups[i] = i;
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
  for (size_t i = 0; i < groups.size(); ++i) {
    uint32_t g = groups[i];
    while (g != groups[g]) g = groups[g];
    groups[i] = g;
  }
  return groups;
}

}  // namespace utils
}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TOFU_UTILS_H_
