#include "memory/cvi_memory_pool.hpp"

#include <cvi_buffer.h>
#include <cvi_vb.h>

#include "cvi_sys.h"
#include "cvi_tdl_log.hpp"
#include "image/vpss_image.hpp"
CviMemoryPool::CviMemoryPool() { CVI_VB_Init(); }

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
// bool CviMemoryPool::allocateImage(std::shared_ptr<BaseImage> &image) {
//   if (image == nullptr) {
//     std::cout << "image is nullptr" << std::endl;
//     return false;
//   }
//   if (image->getImageType() != ImageImplType::VPSS) {
//     std::cout << "image type is not VPSS" << std::endl;
//     return false;
//   }
//   VB_CAL_CONFIG_S stVbCalConfig;

//   COMMON_GetPicBufferConfig(image->getWidth(), image->getHeight(),
//                             image->getImageFormat(), DATA_BITWIDTH_8,
//                             COMPRESS_MODE_NONE, DEFAULT_ALIGN,
//                             &stVbCalConfig);

//   VPSSImage *vpss_image = dynamic_cast<VPSSImage *>(image);
//   VIDEO_FRAME_INFO_S *frame = vpss_image->getFrame();
//   frame->stVFrame.enCompressMode = COMPRESS_MODE_NONE;
//   frame->stVFrame.enPixelFormat = image->getImageFormat();
//   frame->stVFrame.enVideoFormat = VIDEO_FORMAT_LINEAR;
//   frame->stVFrame.enColorGamut = COLOR_GAMUT_BT709;
//   frame->stVFrame.u32Width = image->getWidth();
//   frame->stVFrame.u32Height = image->getHeight();
//   frame->stVFrame.u32Stride[0] = stVbCalConfig.u32MainStride;
//   frame->stVFrame.u32Stride[1] = stVbCalConfig.u32CStride;
//   frame->stVFrame.u32Stride[2] = stVbCalConfig.u32CStride;
//   frame->stVFrame.u32TimeRef = 0;
//   frame->stVFrame.u64PTS = 0;
//   frame->stVFrame.enDynamicRange = DYNAMIC_RANGE_SDR8;

//   std::unique_ptr<MemoryBlock> block = allocate(stVbCalConfig.u32VBSize,
//   1000); if (block == nullptr) {
//     std::cout << "allocate block failed" << std::endl;
//     return false;
//   }

//   //   frame->u32PoolId = block->mem_id;
//   frame->stVFrame.u32Length[0] =
//       ALIGN(stVbCalConfig.u32MainYSize, stVbCalConfig.u16AddrAlign);
//   frame->stVFrame.u32Length[1] = frame->stVFrame.u32Length[2] =
//       ALIGN(stVbCalConfig.u32MainCSize, stVbCalConfig.u16AddrAlign);

//   frame->stVFrame.u64PhyAddr[0] = block->physicalAddress;
//   frame->stVFrame.u64PhyAddr[1] =
//       frame->stVFrame.u64PhyAddr[0] + frame->stVFrame.u32Length[0];
//   frame->stVFrame.u64PhyAddr[2] =
//       frame->stVFrame.u64PhyAddr[1] + frame->stVFrame.u32Length[1];
//   frame->stVFrame.pu8VirAddr[0] = (uint8_t *)block->virtualAddress;
//   frame->stVFrame.pu8VirAddr[1] =
//       frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
//   frame->stVFrame.pu8VirAddr[2] =
//       frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
//   //   image->setMemoryPool(block);
//   return true;
// }

int32_t CviMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
  return 0;
}

int32_t CviMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
  return 0;
}