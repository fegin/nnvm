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

}  // namespace utils
}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_TOFU_UTILS_H_
