/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file geometry.h
 * \brief Data structure used for geometry.
 */
#ifndef NNVM_PASS_GEOMETRY_H_
#define NNVM_PASS_GEOMETRY_H_

#include <nnvm/base.h>
#include <nnvm/scheme.h>

namespace nnvm {
namespace pass {

// TODO(minjie): how to deal with value overflow?
typedef int64_t cost_t;

class Region {
 public:
  // Constructors.
  Region() {}
  Region(const TShape& shp):
    entry_shape_(shp), region_offset_(shp.ndim()), region_shape_(shp) {
    for (size_t i = 0; i < shp.ndim(); ++i) {
      region_offset_[i] = 0;
    }
  }
  Region(const TShape& ent_shp, const TShape& reg_off,
         const TShape& reg_shp):
    entry_shape_(ent_shp), region_offset_(reg_off),
    region_shape_(reg_shp) {}

  inline const TShape& shape() const { return region_shape_; }

  inline const TShape& offset() const { return region_offset_; }

  inline const TShape& entry_shape() const { return entry_shape_; }

  // Partition this region into two sub-regions.
  std::pair<Region, Region> Split2(const Scheme& sch) const;

  // Partition this region into subregions.
  std::vector<Region> Split(const Scheme& sch, size_t k) const;

  // Return true if the region could be splitted using the given scheme.
  bool CanSplit2(const Scheme& sch) const;

  // Area of the region.
  inline cost_t Area() const { return region_shape_.Size(); }

  // Compute the intersection area.
  static cost_t IntersectArea(const Region& r1, const Region& r2);

  // Compute the conversion cost from r1 to r2. The scheme only
  // partitions regions into two parts.
  static cost_t ConvertCost2(const Region& r1, const Scheme& sch1,
                             const Region& r2, const Scheme& sch2);

 private:
  // Shape of the entry this region belongs to.
  TShape entry_shape_;
  // Region offset, and shape.
  TShape region_offset_, region_shape_;
};

std::ostream& operator << (std::ostream& os, const nnvm::pass::Region& region);

}  // namespace pass
}  // namespace nnvm

#endif  // NNVM_PASS_GEOMETRY_H_
