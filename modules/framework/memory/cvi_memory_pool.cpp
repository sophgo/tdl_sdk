#include "memory/cvi_memory_pool.hpp"

#include <cvi_buffer.h>
#include <cvi_vb.h>

#include "cvi_sys.h"
#include "cvi_tdl_log.hpp"
#include "image/vpss_image.hpp"
CviMemoryPool::CviMemoryPool() {}

CviMemoryPool::~CviMemoryPool() {}

std::unique_ptr<MemoryBlock> CviMemoryPool::allocate(uint32_t size,
                                                     uint32_t timeout_ms) {
  VB_POOL_CONFIG_S cfg;
  cfg.u32BlkSize = size;
  cfg.u32BlkCnt = 1;
  cfg.enRemapMode = VB_REMAP_MODE_NONE;
  sprintf(cfg.acName, "%s_%d", str_mem_pool_name_.c_str(), num_allocated_);

  std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();

  CVI_S32 ret =
      CVI_SYS_IonAlloc(&block->physicalAddress, &block->virtualAddress,
                       cfg.acName, cfg.u32BlkSize);
  if (ret != CVI_SUCCESS) {
    std::cout << "allocate ion failed" << std::endl;
    return nullptr;
  }

  block->size = size;
  block->own_memory = true;
  num_allocated_++;

  return block;
}

int32_t CviMemoryPool::release(std::unique_ptr<MemoryBlock> &block) {
  if (block != nullptr && block->own_memory) {
    CVI_SYS_IonFree(block->physicalAddress, block->virtualAddress);
    return 0;
  }
  return -1;
}

std::unique_ptr<MemoryBlock> CviMemoryPool::create_vb(uint32_t size) {
  VB_POOL_CONFIG_S cfg;
  cfg.u32BlkSize = size;
  cfg.u32BlkCnt = 1;
  cfg.enRemapMode = VB_REMAP_MODE_NONE;
  sprintf(cfg.acName, "cvi_vb");
  uint32_t pool_id = CVI_VB_CreatePool(&cfg);
  if (pool_id == VB_INVALID_POOLID) {
    std::cout << "create pool failed" << std::endl;
    return nullptr;
  }
  // std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
  CVI_S32 ret = CVI_VB_MmapPool(pool_id);
  if (ret != CVI_SUCCESS) {
    std::cout << "mmap pool failed" << std::endl;
    return nullptr;
  }
  std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
  VB_BLK blk = CVI_VB_GetBlock(pool_id, size);
  if (blk == (unsigned long)CVI_INVALID_HANDLE) {
    printf("Can't acquire VB block for size %d\n", size);
    return nullptr;
  }

  block->id = pool_id;
  block->physicalAddress = CVI_VB_Handle2PhysAddr(blk);
  ret = CVI_VB_GetBlockVirAddr(pool_id, blk, &block->virtualAddress);
  if (ret != CVI_SUCCESS) {
    std::cout << "get block vir addr failed" << std::endl;
    return nullptr;
  }
  block->size = size;
  return block;
}

int32_t CviMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
  if (block == nullptr || block->virtualAddress == nullptr ||
      block->physicalAddress == 0) {
    LOGI("flushCache block is nullptr");
    return -1;
  }
  CVI_S32 ret = CVI_SYS_IonFlushCache(block->physicalAddress,
                                      block->virtualAddress, block->size);
  LOGI("flushCache done,ret:%d,phyaddr:%lx,viraddr:%lx,size:%d", ret,
       block->physicalAddress, block->virtualAddress, block->size);
  return (int32_t)ret;
}

int32_t CviMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
  if (block == nullptr || block->virtualAddress == nullptr ||
      block->physicalAddress == 0) {
    LOGI("invalidateCache block is nullptr");
    return -1;
  }
  CVI_S32 ret = CVI_SYS_IonInvalidateCache(block->physicalAddress,
                                           block->virtualAddress, block->size);
  LOGI("invalidateCache done,ret:%d,phyaddr:%lx,viraddr:%lx,size:%d", ret,
       block->physicalAddress, block->virtualAddress, block->size);
  return (int32_t)ret;
}