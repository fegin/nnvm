#include "./geometry.h"

using namespace std;

namespace nnvm {
namespace pass {

ostream& operator << (ostream& os, const Region& region) {
  return os << "[" << region.offset()
            << " + " << region.shape()
            << " in: " << region.entry_shape() << "]";
}

bool Region::CanSplit2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    return region_shape_[sch.dim] % 2 == 0;
  case Scheme::kRep:
    return true;
  case Scheme::kRed:
    return false;
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return false;
}

pair<Region, Region> Region::Split2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    {
    TShape shp = region_shape_;
    CHECK_LT(sch.dim, region_shape_.ndim());
    CHECK(shp[sch.dim] % 2 == 0) << "Dimension " << sch.dim << " of size "
      << shp[sch.dim] << " cannot be splitted into two.";
    shp[sch.dim] /= 2;
    TShape offset = region_offset_;
    offset[sch.dim] += shp[sch.dim];
    Region r1(entry_shape_, region_offset_, shp);
    Region r2(entry_shape_, offset, shp);
    return make_pair(r1, r2);
    }
  case Scheme::kRep:
    {
    return make_pair(*this, *this);
    }
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return pair<Region, Region>();
}
  
cost_t Region::IntersectArea(const Region& r1, const Region& r2) {
  const TShape& r1_end = r1.offset() + r1.shape();
  const TShape& r2_end = r2.offset() + r2.shape();
  const TShape& st = max(r1.offset(), r2.offset());
  const TShape& ed = min(r1_end, r2_end);
  cost_t cost = 1;
  for (size_t i = 0; i < st.ndim(); ++i) {
    if (ed[i] <= st[i]) {
      // No intersection.
      return 0;
    } else {
      cost *= ed[i] - st[i];
    }
  }
  return cost;
}

// Note that it is possible that r1 and r2 have different areas. Consider following
// matmult example:
//  - First cut: C x R = red -> R
//  - Second cut: R x r = R
cost_t Region::ConvertCost2(const Region& r1, const Scheme& sch1,
                            const Region& r2, const Scheme& sch2) {
  CHECK_NE(sch2.type, Scheme::kRed)
    << "Reduction scheme is intermediate and could not be used as conversion target";
  cost_t cost = 0;
  if (sch1.type == Scheme::kRed) {
    // Reduction scheme requires special calculation.
    // Note that if source scheme is reduction, the area of source region and target
    // region may be different.
    if (sch2.type == Scheme::kCut) {
      if (!r2.CanSplit2(sch2)) {
        // Cannot split given the scheme. Return a very large cost that is guaranteed to
        // be worse.
        cost = 100 * (r1.Area() + r2.Area());
      } else {
        cost = r1.Area();
      }
    } else if (sch2.type == Scheme::kRep) {
      cost = 2 * r1.Area();
    } else {
      LOG(FATAL) << "Invalid target scheme: " << sch2;
    }
  } else {
    if (sch1.type == Scheme::kRep) {
      // If the source scheme is replication, then all data could be fetched locally.
    } else if (!r1.CanSplit2(sch1) || !r2.CanSplit2(sch2)) {
      // Cannot split given the scheme. Return a very large cost that is guaranteed to
      // be worse.
      cost = 100 * (r1.Area() + r2.Area());
    } else {
      const pair<Region, Region>& r1split = r1.Split2(sch1);
      const pair<Region, Region>& r2split = r2.Split2(sch2);
      cost += Region::IntersectArea(r1split.first, r2split.second);
      cost += Region::IntersectArea(r1split.second, r2split.first);
    }
    if (sch2.type == Scheme::kRep) {
      // If target scheme is replication, extra cost is required to replicate the area
      // that does not overlap with the source one (i.e, r2 - r1).
      cost += r2.Area() - Region::IntersectArea(r1, r2);
    }
  }
  CHECK_GE(cost, 0);
  return cost;
}

}  // namespace pass
}  // namespace nnvm
