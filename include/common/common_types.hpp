#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <cstdint>
enum class ImagePixDataType { INT8 = 0, UINT8, INT16, UINT16, INT32, UINT32, BF16, FLOAT };

enum class ImageFormat {
  GRAY = 0,
  RGB_PLANAR,
  RGB_PACKED,
  BGR_PLANAR,
  BGR_PACKED,
  YUV420SP_UV,    // NV12,size = width * height * 1.5
  YUV420SP_VU,    // NV21,size = width * height * 1.5
  YUV420P_UV,     // I420,size = width * height * 1.5
  YUV420P_VU,     // YV12,size = width * height * 1.5
  YUV422P_UV_16,  // I422_16,size = width * height * 2
  YUV422P_VU_16,  // YV12_16,size = width * height * 2
  YUV422SP_UV,    // NV16,size = width * height * 2
  YUV422SP_VU,    // NV61,size = width * height * 2
  UNKOWN
};

enum class ImageType { VPSS_FRAME = 0, OPENCV_FRAME, FFMPEG_FRAME, BMCV_FRAME, UNKOWN };

struct MemoryBlock {
  uint64_t physicalAddress;  // 内存块的物理地址（仅用于 SoC 场景）
  void* virtualAddress;      // 内存块的虚拟地址
  uint64_t size;             // 内存块的大小（字节数）
  uint32_t id;
  MemoryBlock() {
    physicalAddress = 0;
    virtualAddress = nullptr;
    size = 0;
    id = UINT32_MAX;
  }
};

struct PreprocessParams {
  ImageFormat dstImageFormat;
  ImagePixDataType dstPixDataType;
  int dstWidth;
  int dstHeight;
  int cropX;
  int cropY;
  int cropWidth;
  int cropHeight;
  float mean[3];
  float scale[3];  // Y=X*scale+mean
  bool keepAspectRatio;
};

#endif  // COMMON_TYPES_H