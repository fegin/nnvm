/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file grid.h
 * \brief Data structure for multi-dimensional grid
 */
#ifndef NNVM_PASS_GRID_H_
#define NNVM_PASS_GRID_H_

#include <nnvm/base.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/scheme.h>

namespace nnvm {
namespace pass {

// A structure that represents a subtensor.
struct Block {
  // Index within grid.
  TShape index;
  // Replication index.
  uint32_t replication_id = 0;
  // Device group.
  uint32_t device_group_id = 0;
  // NodeEntry of this block.
  NodeEntry entry;
};

// A structure that represents a tensor that is partitioned as a grid of blocks(subtensors).
class Grid {
 public:
  typedef std::function<void(const Block& from,
                             const TShape& from_shape,
                             const std::vector<Block*>& to,
                             const TShape& to_shape)> SplitFn;
  typedef std::function<void(const std::vector<const Block*>& from,
                             const TShape& from_shape,
                             Block* to,
                             const TShape& to_shape)> ConcatFn;
  // Create a grid from the partition schemes and the given shape. Node entries are NOT initialized.
  //  - If the given scheme vector is empty, the grid will only consist of one block.
  Grid(const TShape& shape, const std::vector<Scheme>& schemes);
  // Create a grid with only one block and initialize it with the given entry.
  Grid(const TShape& shape, const NodeEntry& init_entry);

  const TShape& shape() const { return shape_; }

  const TShape& block_shape() const { return block_shape_; }

  const TShape& num_blocks() const { return num_blocks_; }

  uint32_t num_replicates() const { return num_replicates_; }

  bool replicate_is_reduction() const { return replicate_is_reduction_; }

  size_t TotalNumBlocks() const { return blocks_.size(); }

  Block& BlockAt(size_t i) { return blocks_.at(i); }
  const Block& BlockAt(size_t i) const { return blocks_.at(i); }

  //  -----------------
  //  |   0   |   1   |
  //  -----------------
  //  |   2   |   3   |
  //  -----------------
  //  After OuterSplit(Scheme::Cut(1), 2)
  //  -----------------
  //  | 0 | 1 | 4 | 5 |
  //  -----------------
  //  | 2 | 3 | 6 | 7 |
  //  -----------------
  void OuterSplit(const Scheme& sch, size_t num_splits, SplitFn splitfn = nullptr);

  //  -----------------
  //  |   0   |   1   |
  //  -----------------
  //  |   2   |   3   |
  //  -----------------
  //  After InnerSplit(Scheme::Cut(1), 2)
  //  -----------------
  //  | 0 | 4 | 1 | 5 |
  //  -----------------
  //  | 2 | 6 | 3 | 7 |
  //  -----------------
  void InnerSplit(const Scheme& sch, size_t num_splits, SplitFn splitfn = nullptr);

  // This is the reverse operation of OuterSplit.
  std::pair<Scheme, size_t> OuterConcat(ConcatFn concatfn = nullptr);

  // This is the reverse operation of InnerSplit.
  std::pair<Scheme, size_t> InnerConcat(ConcatFn concatfn = nullptr);

  void CopyFrom(const Grid& other);
  void PrettyPrint() const;

 private:
  std::vector<std::pair<Scheme, size_t>> outer_schemes_;
  std::vector<std::pair<Scheme, size_t>> inner_schemes_;
  // Shape of the tensor represented by this grid.
  TShape shape_;
  // Shape of the blocks. All blocks are of the same shape.
  TShape block_shape_;
  // Number of blocks(subtensors) on each dimension.
  TShape num_blocks_;
  // Number of replicates (or reductions).
  uint32_t num_replicates_ = 1;
  // If true, the replication actually represents tensors to be reduced.
  bool replicate_is_reduction_ = false;

  // A vector representation of all the blocks. It also acts as the map from
  // device_group_id to the block. For example, to get a block under
  // device_group_id=3, just use blocks[3].
  std::vector<Block> blocks_;
};

// Hash value of the given grid index. If the index is a valid one, the value is
// guaranteed to be within range [0, total_num_blocks).
extern size_t BlockIndexHash(const Grid& grid, const TShape& index, uint32_t rep_id);

class ConstGridIndexMap {
 public:
  ConstGridIndexMap(const Grid& grid): grid_(grid) {
    gridindex2block_.resize(grid.TotalNumBlocks(), 0);
    for (size_t i = 0; i < grid.TotalNumBlocks(); ++i) {
      const Block& blk = grid.BlockAt(i);
      const size_t hash = BlockIndexHash(grid, blk.index, blk.replication_id);
      CHECK(hash < gridindex2block_.size());
      gridindex2block_[hash] = i;
    }
  }

  inline const Block& GetBlock(const TShape& index, uint32_t rep_id) const {
    size_t hash = gridindex2block_.at(BlockIndexHash(grid_, index, rep_id));
    return grid_.BlockAt(hash);
  }

 private:
  const Grid& grid_;
  // A map from grid index to the block index in the vector representation.
  // Since the grid index could be mapped to a continuous range from [0, total_num_blocks),
  // a vector could be used here instead of a map.
  std::vector<size_t> gridindex2block_;
};

class GridIndexMap {
 public:
  GridIndexMap(Grid& grid): grid_(grid) {
    gridindex2block_.resize(grid.TotalNumBlocks(), 0);
    for (size_t i = 0; i < grid.TotalNumBlocks(); ++i) {
      const Block& blk = grid.BlockAt(i);
      const size_t hash = BlockIndexHash(grid, blk.index, blk.replication_id);
      CHECK(hash < gridindex2block_.size());
      gridindex2block_[hash] = i;
    }
  }

  inline Block& GetBlock(const TShape& index, uint32_t rep_id) {
    size_t hash = gridindex2block_.at(BlockIndexHash(grid_, index, rep_id));
    return grid_.BlockAt(hash);
  }

 private:
  Grid& grid_;
  // A map from grid index to the block index in the vector representation.
  // Since the grid index could be mapped to a continuous range from [0, total_num_blocks),
  // a vector could be used here instead of a map.
  std::vector<size_t> gridindex2block_;
};

class IndexIter {
 public:
  IndexIter(const TShape& limit): limit_(limit), index_(limit.ndim(), 0) { }

  bool Next() {
    for (int i = index_.ndim() - 1; i >= 0; --i) {
      index_[i] += 1;
      if (index_[i] == limit_[i]) {
        index_[i] = 0;
      } else {
        return true;
      }
    }
    return false;
  }

  const TShape& Get() const { return index_; }

 private:
  // (No ownership).
  const TShape& limit_;
  TShape index_;
};

}  // namespace pass
}  // namespace nnvm
#endif  // NNVM_PASS_GRID_H_
