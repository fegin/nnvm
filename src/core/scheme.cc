#include <nnvm/scheme.h>
#include <nnvm/tuple.h>

using namespace std;

namespace nnvm {

TShape operator + (const TShape& shp1, const TShape& shp2) {
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] += shp2[i];
  }
  return ret;
}

TShape operator - (const TShape& shp1, const TShape& shp2) {
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] -= shp2[i];
  }
  return ret;
}

TShape operator / (const TShape& shp1, const TShape& shp2) {
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    CHECK(shp2[i] != 0 && ret[i] % shp2[i] == 0);
    ret[i] /= shp2[i];
  }
  return ret;
}

TShape max(const TShape& shp1, const TShape& shp2) {
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::max(ret[i], shp2[i]);
  }
  return ret;
}

TShape min(const TShape& shp1, const TShape& shp2) {
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::min(ret[i], shp2[i]);
  }
  return ret;
}

namespace pass {
ostream& operator << (ostream& os, const Scheme& sch) {
  switch (sch.type) {
  case Scheme::kCut: return os << "C" << sch.dim;
  case Scheme::kRep: return os << "Rp";
  case Scheme::kRed: return os << "Rd";
  default:
    LOG(FATAL) << "Unknown scheme type: " << sch.type;
  }
  return os;
}

}  // namespace pass
}  // namespace nnvm

