#include "./grid.h"

#include <iomanip>

using namespace std;

namespace nnvm {
namespace pass {

Grid::Grid(const TShape& shape, const vector<Scheme>& schemes):
  shape_(shape), block_shape_(shape), num_blocks_(shape.ndim()) {
  // Initialize the grid with one block.
  Block blk;
  blk.index = TShape(shape.ndim(), 0);
  blocks_.push_back(std::move(blk));
  for (const Scheme& sch : schemes) {
    if (sch.type == Scheme::kRed) {
      replicate_is_reduction_ = true;
      break;
    }
  }
  // Partition the block using the given schemes.
  // Loop the scheme list in reverse order since this is easier (and more efficient)
  // for grid construction.
  for (int i = schemes.size() - 1; i >= 0; --i) {
    const Scheme& sch = schemes[i];
    OuterSplit(sch, 2);
  }
  CHECK_EQ(num_blocks_.Size() * num_replicates_, blocks_.size());
  // Change the device group id for the blocks.
  for (size_t i = 0; i < blocks_.size(); ++i) {
    blocks_[i].device_group_id = i;
  }
}

Grid::Grid(const TShape& shape, const NodeEntry& init_entry):
  shape_(shape), block_shape_(shape), num_blocks_(shape.ndim()) {
  Block blk;
  blk.index = TShape(shape.ndim(), 0);
  blk.entry = init_entry;
  blocks_.push_back(std::move(blk));
}

void Grid::OuterSplit(const Scheme& sch, size_t num_splits, Grid::SplitFn splitfn) {
  if (num_splits <= 1) {
    return;
  }
  outer_schemes_.push_back(make_pair(sch, num_splits));
  // The new blocks are initialized as replicates of the old blocks.
  // For example, if we have originally 4 blocks and num_splits=3,
  // which means each block will be "splitted"(or replicated) into three new blocks.
  // We will end up with 4*3=12 blocks. The old block vector will be replicated
  // three times.
  const size_t old_num_blks = blocks_.size();
  const TShape old_block_shape = block_shape_;
  vector<Block> new_blocks(num_splits * old_num_blks);
  for (size_t i = 0; i < num_splits; ++i) {
    std::copy(blocks_.begin(), blocks_.end(),
              new_blocks.begin() + i * old_num_blks);
  }
  switch (sch.type) {
  case Scheme::kCut:
    {
      // Change block index.
      const int cut_dim = sch.dim;
      for (size_t i = old_num_blks; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].index[cut_dim] += split_id * num_blocks_[cut_dim];
      }
      num_blocks_[cut_dim] *= num_splits;
      block_shape_[cut_dim] /= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      // Change replication index.
      for (size_t i = old_num_blks; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].replication_id += split_id * num_replicates_;
      }
      num_replicates_ *= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      // Change replication index.
      for (size_t i = old_num_blks; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].replication_id += split_id * num_replicates_;
      }
      num_replicates_ *= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  // If the scheme is Cut and split function is given, split the old blocks
  // to new block vector. If the scheme is Rep, nothing will be done since
  // the block is initialized as replicated.
  if (splitfn && sch.type == Scheme::kCut) {
    // Call splitfn.
    vector<Block*> to(num_splits);
    for (size_t i = 0; i < old_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        to[j] = &new_blocks[i + j * old_num_blks];
      }
      splitfn(blocks_[i], old_block_shape, to, block_shape_);
    }
  }
  blocks_.swap(new_blocks);
}

pair<Scheme, size_t> Grid::OuterConcat(Grid::ConcatFn concatfn) {
  const pair<Scheme, size_t> last = outer_schemes_[outer_schemes_.size() - 1];
  const Scheme& sch = last.first;
  const size_t num_splits = last.second;
  outer_schemes_.pop_back();
  const size_t new_num_blks = blocks_.size() / num_splits;
  const TShape old_block_shape = block_shape_;
  vector<Block> new_blocks(new_num_blks);
  std::copy(blocks_.begin(), blocks_.begin() + new_num_blks, new_blocks.begin());
  switch (sch.type) {
  case Scheme::kCut:
    {
      num_blocks_[sch.dim] /= num_splits;
      block_shape_[sch.dim] *= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      num_replicates_ /= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      num_replicates_ /= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  if (concatfn && sch.type == Scheme::kCut) {
    // Call concatfn.
    vector<const Block*> from(num_splits);
    for (size_t i = 0; i < new_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        from[j] = &blocks_[i + j * new_num_blks];
      }
      concatfn(from, old_block_shape, &new_blocks[i], block_shape_);
    }
  }
  blocks_.swap(new_blocks);
  return last;
}

void Grid::InnerSplit(const Scheme& sch, size_t num_splits, Grid::SplitFn splitfn) {
  if (num_splits <= 1) {
    return;
  }
  inner_schemes_.push_back(make_pair(sch, num_splits));
  // The new blocks are initialized as replicates of the old blocks.
  // For example, if we have originally 4 blocks and num_splits=3,
  // which means each block will be "splitted"(or replicated) into three new blocks.
  // We will end up with 4*3=12 blocks. The old block vector will be replicated
  // three times.
  const size_t old_num_blks = blocks_.size();
  const TShape old_block_shape = block_shape_;
  vector<Block> new_blocks(num_splits * old_num_blks);
  for (size_t i = 0; i < num_splits; ++i) {
    std::copy(blocks_.begin(), blocks_.end(),
              new_blocks.begin() + i * old_num_blks);
  }
  switch (sch.type) {
  case Scheme::kCut:
    {
      // Change block index.
      const int cut_dim = sch.dim;
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].index[cut_dim] *= num_splits;
        new_blocks[i].index[cut_dim] += split_id;
      }
      num_blocks_[cut_dim] *= num_splits;
      block_shape_[cut_dim] /= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      // Change replication index.
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].replication_id *= num_splits;
        new_blocks[i].replication_id += split_id;
      }
      num_replicates_ *= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      // Change replication index.
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        const size_t split_id = i / old_num_blks;
        new_blocks[i].replication_id *= num_splits;
        new_blocks[i].replication_id += split_id;
      }
      num_replicates_ *= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  // If the scheme is Cut and split function is given, split the old blocks
  // to new block vector. If the scheme is Rep, nothing will be done since
  // the block is initialized as replicated.
  if (splitfn && sch.type == Scheme::kCut) {
    // Call splitfn.
    vector<Block*> to(num_splits);
    for (size_t i = 0; i < old_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        to[j] = &new_blocks[i + j * old_num_blks];
      }
      splitfn(blocks_[i], old_block_shape, to, block_shape_);
    }
  }
  blocks_.swap(new_blocks);
}

pair<Scheme, size_t> Grid::InnerConcat(Grid::ConcatFn concatfn) {
  const pair<Scheme, size_t> last = inner_schemes_[inner_schemes_.size() - 1];
  const Scheme& sch = last.first;
  const size_t num_splits = last.second;
  inner_schemes_.pop_back();
  const size_t new_num_blks = blocks_.size() / num_splits;
  const TShape old_block_shape = block_shape_;
  vector<Block> new_blocks(new_num_blks);
  std::copy(blocks_.begin(), blocks_.begin() + new_num_blks, new_blocks.begin());
  switch (sch.type) {
  case Scheme::kCut:
    {
      // Change block index.
      const int cut_dim = sch.dim;
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        new_blocks[i].index[cut_dim] /= num_splits;
      }
      num_blocks_[sch.dim] /= num_splits;
      block_shape_[sch.dim] *= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      // Change replication index.
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        new_blocks[i].replication_id /= num_splits;
      }
      num_replicates_ /= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      // Change replication index.
      for (size_t i = 0; i < new_blocks.size(); ++i) {
        new_blocks[i].replication_id /= num_splits;
      }
      num_replicates_ /= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  if (concatfn && sch.type == Scheme::kCut) {
    // Call concatfn.
    vector<const Block*> from(num_splits);
    for (size_t i = 0; i < new_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        from[j] = &blocks_[i + j * new_num_blks];
      }
      concatfn(from, old_block_shape, &new_blocks[i], block_shape_);
    }
  }
  blocks_.swap(new_blocks);
  return last;
}

void Grid::CopyFrom(const Grid& other) {
  CHECK(this->num_blocks() == other.num_blocks());
  CHECK(this->num_replicates() == other.num_replicates());
  GridIndexMap this_idx(*this);
  ConstGridIndexMap other_idx(other);
  IndexIter iter(this->num_blocks());
  do {
    const TShape& curidx = iter.Get();
    for (size_t repid = 0; repid < other.num_replicates(); ++repid) {
      const Block& from_block = other_idx.GetBlock(curidx, repid);
      Block& to_block = this_idx.GetBlock(curidx, repid);
      to_block.entry = from_block.entry;
    }
  } while(iter.Next());
}

void Grid::PrettyPrint() const {
  // Only pretty print 2D.
  if (num_blocks_.ndim() <= 1) {
    LOG(INFO) << "Only an 1D block.";
  }
  ConstGridIndexMap grididx(*this);
  std::ostringstream oss;
  oss << "+";
  for (size_t j = 0; j < num_blocks_[1]; ++j) {
    oss << "----+";
  }
  oss << "\n";
  TShape idx(num_blocks_.ndim(), 0);
  for (size_t i = 0; i < num_blocks_[0]; ++i) {
    oss << "|";
    for (size_t j = 0; j < num_blocks_[1]; ++j) {
      idx[0] = i;
      idx[1] = j;
      const uint32_t dev = grididx.GetBlock(idx, 0).device_group_id;
      oss << std::setw(3) << dev << " |";
    }
    oss << "\n+";
    for (size_t j = 0; j < num_blocks_[1]; ++j) {
      oss << "----+";
    }
    oss << "\n";
  }
  cout << oss.str() << endl;
}

size_t BlockIndexHash(const Grid& grid, const TShape& index, uint32_t rep_id) {
  CHECK_EQ(index.ndim(), grid.shape().ndim()) << grid.shape() << " " << index;
  CHECK_LT(rep_id, grid.num_replicates());
  size_t hash = 0, mult = 1;
  for (int i = index.ndim() - 1; i >= 0; --i) {
    CHECK_LT(index[i], grid.num_blocks()[i]);
    hash += index[i] * mult;
    mult *= grid.num_blocks()[i];
  }
  hash += rep_id * mult;
  return hash;
}

}  // namespace pass
}  // namespace nnvm
