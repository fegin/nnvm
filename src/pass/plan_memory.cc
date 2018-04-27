/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/scheme.h>
#include <memory>
#include "./graph_algorithm.h"
#include "./tofu/utils.h"

namespace nnvm {
namespace pass {
namespace {
inline bool StartsWith(const std::string& value, const std::string& starting) {
  if (starting.size() > value.size()) return false;
  return std::equal(starting.begin(), starting.end(), value.begin());
}

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int64_t;
  // bad storage id
  static const StorageID kBadStorageID = -1;
  // external storage id
  static const StorageID kExternalStorageID = -2;
  // Tofu temporary storage
  static const StorageID kTofuTempStorageID = -3;
  // Tofu persistent storage
  static const StorageID kTofuPersistStorageID = -4;
  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) {
      return kBadStorageID;
    }
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size, node_color_[node_id]);
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          e->color != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          e->color != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size, node_color_[node_id]);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    if (id == kExternalStorageID) return;
    StorageEntry *e = data_[id].get();
    free_.insert({e->max_bytes, e});
  }
  size_t Bytes(StorageID sid) const {
    if (sid < 0 || sid >= data_.size()) return 0;
    return data_[sid]->max_bytes;
  }
  
  int DeviceId(StorageID sid) const {
    if (sid < 0 || sid >= data_.size()) return 0;
    return data_[sid]->device_id;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx,
                          const DeviceVector& devices)
    : idx_(idx) {
    this->Init(dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16),
               dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1),
               devices);
  }

 private:
  // initialize the graph allocator
  void Init(size_t match_range, uint32_t num_match_color,
      const DeviceVector& devices) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    std::vector<uint32_t> importance(idx_->num_nodes(), 0);
    for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
      const Node* node = (*idx_)[nid].source;
      if (node->is_variable()) {
        continue;
      }
      importance[nid] = 1;
    }
    num_match_color_ = pass::ColorNodeGroup(
        *idx_, importance, num_match_color_, &node_color_);
    //bool has_extra_color = false;
    for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
      // TOFU's node and copy nodes have different colors than others.
      const Node* node = (*idx_)[nid].source;
      if (StartsWith(node->attrs.name, "_TOFU")
          || node->op() == Op::Get("_CrossDeviceCopy")) {
        node_color_[nid] = num_match_color_++;
        //has_extra_color = true;
      }
    }
    //if (has_extra_color) ++num_match_color_;
    //for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
      //const Node* node = (*idx_)[nid].source;
      //LOG(INFO) << "Node#" << nid << ": "
        //<< node->attrs.name << " color: " << node_color_[nid];
    //}
  }

  StorageID Alloc(int dev_id, size_t size, uint32_t color) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    ptr->color = color;
    data_.emplace_back(std::move(ptr));
    return id;
  }

  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    //uint32_t released_by_node{0};
    uint32_t color{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

// function to plan memory
Graph PlanMemoryGraphColoring(Graph ret) {
  const int tofu_cache_temp_memory = dmlc::GetEnv("TOFU_CACHE_TEMP_MEMORY", 0);
  // setup ref counter
  const IndexedGraph& idx = ret.indexed_graph();

  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  // reference counter of each node
  std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
  // step 1: initialize reference count
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    for (const auto& e : inode.inputs) {
      ++ref_count[idx.entry_id(e)];
    }
    // no dataflow dependency is needed for those are ignored.
    // revoke the dependency counter.
    if (fignore_inputs.count(inode.source->op()) != 0) {
      auto ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      for (uint32_t i : ignore_inputs) {
        --ref_count[idx.entry_id(inode.inputs[i])];
      }
    }
  }
  for (const auto& e : idx.outputs()) {
    ++ref_count[idx.entry_id(e)];
  }
  // step 2: allocate memory.
  StorageVector storage;

  if (ret.attrs.count("storage") != 0) {
    storage = ret.MoveCopyAttr<StorageVector>("storage");
  } else {
    storage.resize(idx.num_node_entries(), -1);
  }

  std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);
  const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
  const DeviceVector& device_vec = ret.GetAttr<DeviceVector>("device");
  static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
  CHECK_EQ(storage.size(), idx.num_node_entries());
  CHECK_EQ(shape_vec.size(), idx.num_node_entries());
  CHECK_EQ(dtype_vec.size(), idx.num_node_entries());
  CHECK_EQ(device_vec.size(), idx.num_nodes());

  // the allocator.
  GraphAllocator allocator(&idx, device_vec);
  // number of entries that are not statically allocated.
  size_t num_not_allocated = 0;

  LOG(INFO) << "#Nodes=" << idx.num_nodes() << " #Entries=" << idx.num_node_entries();
  // Process some tofu generated nodes.
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    if (inode.source->op() == Op::Get("_TofuFakeVar")) {
      // Note: TofuFakeVar is a special source node created to represent the split
      // of the variable that are distributed across devices. Its output must be persisted.
      CHECK_EQ(inode.source->num_outputs(), 1);
      uint32_t eid = idx.entry_id(nid, 0);
      storage[eid] = GraphAllocator::kTofuPersistStorageID;
    } else if (inode.source->op() == Op::Get("_TofuFakeOut")) {
      // Note: TofuFakeOut is a special sink node created to represent the concatenation
      // of data that are distributed across devices. Its inputs cannot be freed.
      const Node* node = inode.source;
      CHECK_EQ(node->num_outputs(), 1);
      uint32_t eid = idx.entry_id(nid, 0);
      const auto& dep_ent_idx = nnvm::get<std::vector<uint32_t>>(node->attrs.parsed);
      CHECK_EQ(dep_ent_idx.size(), node->control_deps.size());
      for (size_t i = 0; i < dep_ent_idx.size(); ++i) {
        uint32_t eid = idx.entry_id(idx.node_id(node->control_deps[i].get()), dep_ent_idx[i]);
        storage[eid] = GraphAllocator::kTofuPersistStorageID;
      }
      storage[eid] = GraphAllocator::kExternalStorageID;
    }
  }
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    //LOG(INFO) << "Plan for node#" << nid << " " << inode.source->attrs.name;
    if (tofu_cache_temp_memory &&
        utils::StartsWith(inode.source->attrs.name, "_TOFU_CONVERT")) {
      const Node* node = inode.source;
      const auto& param = nnvm::get<pass::TofuConvertParam>(node->attrs.parsed);
      if (param.input_or_output) {
        // Input conversion. The conversion output is temporary space and should not
        // be allocated.
        CHECK_EQ(node->num_outputs(), 1);
        uint32_t eid = idx.entry_id(nid, 0);
        storage[eid] = GraphAllocator::kTofuTempStorageID;
        continue;
      } else {
        // Output conversion. The conversion input is temporary space and should not
        // be allocated. The output need to be allocated as normal.
        for (const auto& ent : node->inputs) {
          uint32_t eid = idx.entry_id(ent);
          storage[eid] = GraphAllocator::kTofuTempStorageID;
        }
      }
    }
    // check inplace option
    if (finplace_option.count(inode.source->op()) != 0) {
      auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
      for (auto& kv : inplace_pairs) {
        uint32_t eid_out = idx.entry_id(nid, kv.second);
        uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
        if (ref_count[eid_in] == 1 &&
            ref_count[eid_out] != 0 &&
            storage[eid_out] == GraphAllocator::kBadStorageID &&
            storage[eid_in] != GraphAllocator::kBadStorageID &&
            shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
            dtype_vec[eid_out] == dtype_vec[eid_in]) {
          // inplace optimization
          storage[eid_out] = storage[eid_in];
          ref_count[eid_in] = 0;
          storage_inplace_index[eid_out] = kv.first;
        }
      }
    }
    // normal allocation
    const int dev_id = device_vec[nid];
    // allocate output
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        storage[eid] = allocator.Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
      }
    }

    // check if certain inputs is ignored.
    std::vector<uint32_t> ignore_inputs;
    if (fignore_inputs.count(inode.source->op()) != 0) {
      ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      std::sort(ignore_inputs.begin(), ignore_inputs.end());
    }
    // then free inputs.
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      // ref counter of ignored input is already decreased.
      if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
      const auto& e = inode.inputs[i];
      uint32_t eid = idx.entry_id(e);
      // temp_ref_count == 0 means it is taken by inplace op
      if (ref_count[eid] == 0) continue;
      // if we decrease it to zero, means we are ready to relase
      --ref_count[eid];
      if (ref_count[eid] == 0 && storage[eid] >= 0) {
        allocator.Release(storage[eid], nid);
      }
    }
    // check if there are outputs that can be freeded immediately
    // these output are not referenced by any operator.
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      if (ref_count[eid] == 0 && storage[eid] >= 0) {
        allocator.Release(storage[eid], nid);
        // use -2 to indicate that the node was never touched.
        storage_inplace_index[eid] = -2;
      }
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        ++num_not_allocated;
      }
    }
  }

  std::map<int, std::vector<std::pair<uint32_t, size_t>>> storage2entry;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const auto& inode = idx[nid];
    for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
      uint32_t eid = idx.entry_id(nid, i);
      //LOG(INFO) << "Storage for Node#" << nid << " " << inode.source->attrs.name
        //<< "#" << i << ": " << storage[eid];
      storage2entry[storage[eid]].push_back(std::make_pair(nid, i));
    }
  }
  size_t total_alloc_bytes = 0;
  size_t bad_alloc_bytes = 0, extern_alloc_bytes = 0;
  size_t temp_alloc_bytes = 0, persist_alloc_bytes = 0;
  std::vector<size_t> dev2bytes(100, 0);
  for (const auto& kv : storage2entry) {
    LOG(INFO) << "Storage#" << kv.first << ": [";
    size_t size_sum = 0;
    for (const auto& e : kv.second) {
      LOG(INFO) << "\t" << idx[e.first].source->attrs.name << "#" << e.second;
      const uint32_t eid = idx.entry_id(e.first, e.second);
      size_sum += shape_vec[eid].Size();
    }
    size_sum *= 4;
    size_t bytes = allocator.Bytes(kv.first);
    int dev = allocator.DeviceId(kv.first);
    LOG(INFO) << "] bytes=" << bytes << " device=" << dev;
    if (kv.first == GraphAllocator::kBadStorageID) {
      bad_alloc_bytes += size_sum;
    } else if (kv.first == GraphAllocator::kExternalStorageID) {
      extern_alloc_bytes += size_sum;
    } else if (kv.first == GraphAllocator::kTofuTempStorageID) {
      temp_alloc_bytes += size_sum;
    } else if (kv.first == GraphAllocator::kTofuPersistStorageID) {
      persist_alloc_bytes += size_sum;
    } else {
      total_alloc_bytes += bytes;
      dev2bytes.at(dev) += bytes;
    }
  }
  LOG(INFO) << "Total allocated bytes: " << total_alloc_bytes;
  LOG(INFO) << "Total allocated bytes(device#0): " << dev2bytes.at(0);
  LOG(INFO) << "Total allocated bytes(device#1): " << dev2bytes.at(1);
  LOG(INFO) << "Total allocated bytes(device#2): " << dev2bytes.at(2);
  LOG(INFO) << "Total allocated bytes(device#3): " << dev2bytes.at(3);
  LOG(INFO) << "Total bad alloc bytes: " << bad_alloc_bytes;
  LOG(INFO) << "Total extern alloc bytes: " << extern_alloc_bytes;
  LOG(INFO) << "Total tofu temp alloc bytes: " << temp_alloc_bytes;
  LOG(INFO) << "Total tofu persist alloc bytes: " << persist_alloc_bytes;
  LOG(INFO) << "Total memory buckets: " << storage2entry.size() - 3;

  ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage));
  ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
  ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(total_alloc_bytes);
  ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(num_not_allocated);
  return ret;
}

Graph PlanMemory(Graph ret) {
  return PlanMemoryGraphColoring(ret);
}

NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");

}  // namespace
}  // namespace pass
}  // namespace nnvm
