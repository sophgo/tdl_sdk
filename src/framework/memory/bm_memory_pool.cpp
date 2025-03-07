#include "memory/bm_memory_pool.hpp"

#include <bmlib_runtime.h>

#include "utils/tdl_log.hpp"
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

  block->size = size;

  unsigned long long addr;
  bm_mem_mmap_device_mem((bm_handle_t)bm_handle_, (bm_device_mem_t *)p_dev,
                         &addr);
  block->virtualAddress = (uint8_t *)addr;
  block->physicalAddress = bm_mem_get_device_addr(*(bm_device_mem_t *)p_dev);
  block->id = 0;
  block->handle = p_dev;
  LOGI("allocate bm memory block,size:%d,phy_addr:%p,virtual_addr:%p", size,
       (void *)block->physicalAddress, (void *)block->virtualAddress);
  return block;
}

int32_t BmMemoryPool::release(std::unique_ptr<MemoryBlock> &block) {
  // TODO: implement
  LOGI("start to release bm memory block,size:%d,phy_addr:%p,virtual_addr:%p",
       block->size, (void *)block->physicalAddress,
       (void *)block->virtualAddress);
  if (block->virtualAddress != nullptr) {
    bm_mem_unmap_device_mem((bm_handle_t)bm_handle_, block->virtualAddress,
                            block->size);
  }
  bm_device_mem_t *p_dev = (bm_device_mem_t *)block->handle;

  uint64_t phy_addr = bm_mem_get_device_addr(*p_dev);
  LOGI("release bm memory block,size:%d,phy_addr:%p,block phy_addr:%p",
       block->size, (void *)phy_addr, (void *)block->physicalAddress);
  bm_free_device(bm_handle_t(bm_handle_), *p_dev);
  delete p_dev;
  return 0;
}

int32_t BmMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
  // TODO: implement
  if (block == nullptr) {
    LOGI("flushCache block is nullptr");
    return -1;
  } else if (block->virtualAddress == nullptr) {
    LOGI("flushCache block->virtualAddress is nullptr");
    return -1;
  } else if (block->physicalAddress == 0) {
    LOGI("flushCache block->physicalAddress is 0");
    return -1;
  }
  bm_status_t st = bm_mem_flush_device_mem((bm_handle_t)bm_handle_,
                                           (bm_device_mem_t *)block->handle);
  if (st != BM_SUCCESS) {
    return -1;
  }
  return 0;
}

int32_t BmMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
  if (block == nullptr) {
    LOGI("invalidateCache block is nullptr");
    return -1;
  } else if (block->virtualAddress == nullptr) {
    LOGI("invalidateCache block->virtualAddress is nullptr");
    return -1;
  } else if (block->physicalAddress == 0) {
    LOGI("invalidateCache block->physicalAddress is 0");
    return -1;
  }
  bm_status_t st = bm_mem_invalidate_device_mem(
      (bm_handle_t)bm_handle_, (bm_device_mem_t *)block->handle);
  if (st != BM_SUCCESS) {
    return -1;
  }
  return 0;
}
