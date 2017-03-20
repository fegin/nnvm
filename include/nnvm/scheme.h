/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */
#ifndef NNVM_PASS_SCHEME_H_
#define NNVM_PASS_SCHEME_H_

#include <vector>
#include <string>
#include <functional>
#include <nnvm/base.h>
#include <nnvm/tuple.h>
#include <nnvm/node.h>

namespace nnvm {
namespace pass {

// Attributes used in operator partition.
struct Scheme {
  enum SchemeType {
    kCut = 0,
    kRep,
    kRed,
  };
  // Scheme type.
  SchemeType type = kRep;
  // If type == kCut, this will be used to indicate which dimension is required
  // to be splitted. Otherwise, this is equal to -1.
  int dim = -1;

  Scheme() {}
  Scheme(SchemeType type, int dim): type(type), dim(dim) {}
  // Creator for cut on given dimension.
  static Scheme Cut(int dim) {
    return Scheme(kCut, dim);
  }
  static Scheme Rep() {
    return Scheme(kRep, -1);
  }
  static Scheme Red() {
    return Scheme(kRed, -1);
  }
};

using FPartition = std::function<NodeAttrs(
    const NodeAttrs& attrs, size_t num_partitions)>;

struct SchemeRequest {
  std::vector<Scheme> input_schemes;
  std::vector<Scheme> output_schemes;
  FPartition partitioner;
};

using FAlignedSchemes = std::function<std::vector<SchemeRequest>(
    const NodeAttrs& attrs,
    const std::vector<TShape>& input_shapes,
    const std::vector<TShape>& output_shapes)>;

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_SCHEME_H_
