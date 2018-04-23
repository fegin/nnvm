#include "./tiling.h"
#include "./tiling_kcuts.h"
#include "./tiling_manual.h"
#include "./tiling_spartan.h"

using namespace std;

namespace nnvm {
namespace pass {

unique_ptr<Tiling> Tiling::Create(
    const string& name,
    Graph* graph,
    const Levels& levels,
    const NodeEntryGroups& eg,
    const NodeGroups& ng,
    size_t num_devices) {
  if (name == "kcuts") {
    return unique_ptr<Tiling>(new CutAlgorithm(graph, levels, eg, ng, num_devices));
  } else if (name == "k-equal-cuts") {
    return unique_ptr<Tiling>(new CutAlgorithm(graph, levels, eg, ng, num_devices, true));
  //} else if (name == "datapar") {
  //  return unique_ptr<Tiling>(new DataParallelism(graph, eg, ng, num_devices));
  //} else if (name == "modelpar") {
  //  return unique_ptr<Tiling>(new ModelParallelism(graph, eg, ng, num_devices));
  //} else if (name == "hybridpar") {
  //  return unique_ptr<Tiling>(new HybridParallelism(graph, eg, ng, num_devices));
  //} else if (name == "spartan") {
  //  return unique_ptr<Tiling>(new SpartanTiling(graph, eg, ng, num_devices));
  //} else if (name == "usertiling") {
  //  return unique_ptr<Tiling>(new UserTiling(graph, eg, ng, num_devices));
  } else {
    LOG(FATAL) << "Unknown tiling: " << name;
  }
  return nullptr;
}

}  // namespace pass
}  // namespace nnvm
