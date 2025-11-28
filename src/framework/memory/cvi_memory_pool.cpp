#include "memory/cvi_memory_pool.hpp"

#include <cvi_buffer.h>
#include <cvi_vb.h>

#include "cvi_sys.h"
#include "image/vpss_image.hpp"
#include "utils/tdl_log.hpp"
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
      CVI_SYS_IonAlloc(reinterpret_cast<CVI_U64 *>(&block->physicalAddress),
                       &block->virtualAddress, cfg.acName, cfg.u32BlkSize);
  if (ret != CVI_SUCCESS) {
    std::cout << "allocate ion failed" << std::endl;
    return nullptr;
  }

  block->size = size;
  block->own_memory = true;
  num_allocated_++;

  LOGI(
      "allocate memory success,size: %d,physicalAddress: %lx,virtualAddress: "
      "%p",
      size, block->physicalAddress, block->virtualAddress);
  return block;
}

int32_t CviMemoryPool::release(std::unique_ptr<MemoryBlock> &block) {
  if (block != nullptr && block->own_memory && block->own_memory == true) {
    CVI_SYS_IonFree(block->physicalAddress, block->virtualAddress);
    return 0;
  }
  block = nullptr;
  return -1;
}

std::unique_ptr<MemoryBlock> CviMemoryPool::CreateExVb(uint32_t blk_cnt,
                                                       uint32_t width,
                                                       uint32_t height,
                                                       void *fmt) {
  VB_POOL_CONFIG_EX_S stExconfig;

  std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
  PIXEL_FORMAT_E pix_fmt = *(PIXEL_FORMAT_E *)fmt;
  width = ALIGN(width, DEFAULT_ALIGN);
  height = ALIGN(height, DEFAULT_ALIGN);
  uint32_t blk_size =
      COMMON_GetPicBufferSize(width, height, pix_fmt, DATA_BITWIDTH_8,
                              COMPRESS_MODE_NONE, DEFAULT_ALIGN);
  char exvb_name[32];
  sprintf(exvb_name, "%s_exvb", str_mem_pool_name_.c_str());
  CVI_S32 ret =
      CVI_SYS_IonAlloc(reinterpret_cast<CVI_U64 *>(&block->physicalAddress),
                       &block->virtualAddress, exvb_name, blk_size * blk_cnt);
  if (ret != CVI_SUCCESS) {
    std::cout << "CVI_SYS_IonAlloc failed! Mem size: " << blk_size * blk_cnt
              << std::endl;
    return nullptr;
  }

  memset(&stExconfig, 0, sizeof(stExconfig));
  stExconfig.u32BlkCnt = blk_cnt;
  for (uint32_t i = 0; i < blk_cnt; i++) {
    uint64_t base = block->physicalAddress + i * blk_size;
    if (pix_fmt == PIXEL_FORMAT_NV21 || pix_fmt == PIXEL_FORMAT_NV12) {
      stExconfig.astUserBlk[i].au64PhyAddr[0] = base;                   // Y
      stExconfig.astUserBlk[i].au64PhyAddr[1] = base + width * height;  // VU
      stExconfig.astUserBlk[i].au64PhyAddr[2] = 0;
    } else if (pix_fmt == PIXEL_FORMAT_RGB_888 ||
               pix_fmt == PIXEL_FORMAT_BGR_888 ||
               pix_fmt == PIXEL_FORMAT_YUV_400) {
      stExconfig.astUserBlk[i].au64PhyAddr[0] = base;  // Packed RGB or GRAY
      stExconfig.astUserBlk[i].au64PhyAddr[1] = 0;
      stExconfig.astUserBlk[i].au64PhyAddr[2] = 0;
    } else if (pix_fmt == PIXEL_FORMAT_RGB_888_PLANAR ||
               pix_fmt == PIXEL_FORMAT_BGR_888_PLANAR) {
      stExconfig.astUserBlk[i].au64PhyAddr[0] = base;                       // R
      stExconfig.astUserBlk[i].au64PhyAddr[1] = base + width * height;      // G
      stExconfig.astUserBlk[i].au64PhyAddr[2] = base + width * height * 2;  // B
    } else {
      LOGE("imageFormat not support, imageFormat: %d", (int32_t)pix_fmt);
      return nullptr;
    }
  }

  VB_POOL pool = CVI_VB_CreateExPool(&stExconfig);
  if (pool == VB_INVALID_POOLID) {
    LOGE(
        "CVI_VB_CreateExPool failed! blk_size: %d, blk_cnt: %d, width: %d, "
        "height: %d",
        blk_size, blk_cnt, width, height);
    CVI_SYS_IonFree(block->physicalAddress, block->virtualAddress);
    return nullptr;
  }
  block->size = blk_size * blk_cnt;
  block->id = pool;
  block->own_memory = true;

  LOGI(
      "allocate memory success,size: %d,physicalAddress: %lx,virtualAddress: "
      "%p",
      blk_size * blk_cnt, block->physicalAddress, block->virtualAddress);
  return block;
}

int32_t CviMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
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
  CVI_S32 ret = CVI_SYS_IonFlushCache(block->physicalAddress,
                                      block->virtualAddress, block->size);
  LOGI("flushCache done,ret:%d,phyaddr:%#llx,viraddr:%lx,size:%d", ret,
       block->physicalAddress, block->virtualAddress, block->size);
  return (int32_t)ret;
}

int32_t CviMemoryPool::DestroyExVb(std::unique_ptr<MemoryBlock> &block) {
  if (block != nullptr && block->own_memory && block->own_memory == true) {
    CVI_VB_DestroyPool(block->id);
    CVI_SYS_IonFree(block->physicalAddress, block->virtualAddress);
    return 0;
  }
  block = nullptr;
  return 0;
}

int32_t CviMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
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
  CVI_S32 ret = CVI_SYS_IonInvalidateCache(block->physicalAddress,
                                           block->virtualAddress, block->size);
  LOGI("invalidateCache done,ret:%d,phyaddr:%#llx,viraddr:%lx,size:%d", ret,
       block->physicalAddress, block->virtualAddress, block->size);
  return (int32_t)ret;
}