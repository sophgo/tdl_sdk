#include "memory/bm_memory_pool.hpp"

#include <bmlib_runtime.h>

#include "cvi_tdl_log.hpp"
BMContext::BMContext() {}

BMContext::~BMContext() {
  for (auto kv : device_handles_) {
    bm_dev_free(kv.second);
  }
  device_handles_.clear();
}
BMContext &BMContext::Get() {
  static BMContext bm_ctx;
  return bm_ctx;
}

bm_handle_t BMContext::get_handle(int device_id) {
  if (device_handles_.count(device_id)) {
    return device_handles_[device_id];
  }

  if (device_id == -1) {
    if (device_handles_.size()) {
      for (auto kv : device_handles_) {
        return kv.second;
      }
    }
    return nullptr;
  }
  pthread_mutex_lock(&lock_);
  bm_handle_t h;
  bm_dev_request(&h, device_id);
  device_handles_[device_id] = h;
  pthread_mutex_unlock(&lock_);
  return device_handles_[device_id];
}

void BMContext::set_device_id(int device_id) {
  // BMContext& inst = Get();
  cnn_bm168x_handle(device_id);
}

BmMemoryPool::BmMemoryPool(void *bm_handle) {
  if (bm_handle == nullptr) {
    bm_handle_ = BMContext::cnn_bm168x_handle(0);  // TODO:specify device id
  } else {
    bm_handle_ = bm_handle;
  }
}
BmMemoryPool::~BmMemoryPool() {}
std::unique_ptr<MemoryBlock> BmMemoryPool::allocate(uint32_t size,
                                                    uint32_t timeout_ms) {
  bm_device_mem_t *p_dev = new bm_device_mem_t();
  bm_status_t st = bm_malloc_device_byte(bm_handle_t(bm_handle_), p_dev, size);
  if (st != BM_SUCCESS) {
    return nullptr;
  }

  std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
  block->handle = p_dev;
  block->size = size;

  unsigned long long addr;
  bm_mem_mmap_device_mem((bm_handle_t)bm_handle_, (bm_device_mem_t *)p_dev,
                         &addr);
  block->virtualAddress = (uint8_t *)addr;
  block->physicalAddress = bm_mem_get_device_addr(*(bm_device_mem_t *)p_dev);
  block->id = 0;
  block->handle = p_dev;
  LOGI("allocate bm memory block,size:%d,phy_addr:%llu,virtual_addr:%p", size,
       block->physicalAddress, block->virtualAddress);
  return block;
}

int32_t BmMemoryPool::release(std::unique_ptr<MemoryBlock> &block) {
  // TODO: implement
  if (block->virtualAddress != nullptr) {
    bm_mem_unmap_device_mem((bm_handle_t)bm_handle_, block->virtualAddress,
                            block->size);
  }
  bm_device_mem_t *p_dev = (bm_device_mem_t *)block->handle;
  bm_free_device(bm_handle_t(bm_handle_), *p_dev);
  delete p_dev;
  return 0;
}

int32_t BmMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
  // TODO: implement
  if (block->virtualAddress != nullptr) {
    bm_status_t st = bm_mem_flush_device_mem((bm_handle_t)bm_handle_,
                                             (bm_device_mem_t *)block->handle);
    if (st != BM_SUCCESS) {
      return -1;
    }
  } else {
    return -1;
  }
  return 0;
}

int32_t BmMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
  // TODO: implement
  if (block->virtualAddress != nullptr) {
    bm_status_t st = bm_mem_invalidate_device_mem(
        (bm_handle_t)bm_handle_, (bm_device_mem_t *)block->handle);
    if (st != BM_SUCCESS) {
      return -1;
    }
  }
  return 0;
}
