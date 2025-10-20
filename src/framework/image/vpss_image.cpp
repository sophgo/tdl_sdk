#include "image/vpss_image.hpp"

#include <cvi_buffer.h>

#include "cvi_vb.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "utils/tdl_log.hpp"

#define SCALAR_4096_ALIGN_BUG 0x1000
VPSSImage::VPSSImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
                     TDLDataType dataType, bool alloc_memory,
                     std::shared_ptr<BaseMemoryPool> memory_pool) {
  int32_t ret = initFrameInfo(width, height, imageFormat, dataType, &frame_);
  if (ret != 0) {
    throw std::runtime_error("initFrameInfo failed");
  }

  // VIDEO_FRAME_S* vFrame = &frame_.stVFrame;
  // CVI_U32 u32MapSize =
  //     vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];
  if (memory_pool == nullptr) {
    memory_pool_ = MemoryPoolFactory::createMemoryPool();
  } else {
    memory_pool_ = memory_pool;
  }
  if (memory_pool_ == nullptr) {
    LOGE("memory_pool_ is nullptr");
    throw std::runtime_error("memory_pool_ is nullptr");
  }
  image_format_ = imageFormat;
  pix_data_type_ = dataType;
  image_type_ = ImageType::VPSS_FRAME;

  if (alloc_memory) {
    int32_t ret = allocateMemory();
    if (ret != 0) {
      LOGE("allocateMemory failed");
      throw std::runtime_error("allocateMemory failed");
    }
  }
  LOGI(
      "VPSSImage constructor "
      "done,width:%d,height:%d,imageFormat:%d,dataType:%d,stride:%d,%d,%d,pix_"
      "format:%d",
      width, height, imageFormat, dataType, frame_.stVFrame.u32Stride[0],
      frame_.stVFrame.u32Stride[1], frame_.stVFrame.u32Stride[2],
      frame_.stVFrame.enPixelFormat);
}

VPSSImage::VPSSImage(const VIDEO_FRAME_INFO_S& frame) {
  frame_ = frame;

  memory_pool_ = MemoryPoolFactory::createMemoryPool();

  int32_t ret = extractImageInfo(frame);
  if (ret != 0) {
    LOGE("extractImageInfo failed, ret: %d", ret);
    throw std::runtime_error("extractImageInfo failed");
  }
}

VPSSImage::~VPSSImage() {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr");
    return;
  }
  LOGI(
      "VPSSImage::~VPSSImage "
      "own_memory:%d,phyaddr:%#llx,viraddr:%lx,width:%d,"
      "height:%d,format:%d,pix_data_type:%d",
      memory_block_->own_memory, frame_.stVFrame.u64PhyAddr[0],
      frame_.stVFrame.pu8VirAddr[0], frame_.stVFrame.u32Width,
      frame_.stVFrame.u32Height, (int)image_format_, (int)pix_data_type_);
  if (memory_block_ != nullptr && memory_block_->own_memory) {
    memory_pool_->release(memory_block_);
    memory_block_ = nullptr;
  }

  memset(&frame_, 0, sizeof(VIDEO_FRAME_INFO_S));
}

int32_t VPSSImage::prepareImageInfo(uint32_t width, uint32_t height,
                                    ImageFormat imageFormat,
                                    TDLDataType pix_data_type,
                                    uint32_t align_size) {
  UNUSED(align_size);
  return initFrameInfo(width, height, imageFormat, pix_data_type, &frame_);
}

bool VPSSImage::isInitialized() const {
  bool is_initialized = frame_.stVFrame.u64PhyAddr[0] != 0 &&
                        frame_.stVFrame.pu8VirAddr[0] != nullptr &&
                        frame_.stVFrame.u32Width != 0 &&
                        frame_.stVFrame.u32Height != 0 &&
                        memory_block_ != nullptr;
  return is_initialized;
}

int32_t VPSSImage::initFrameInfo(uint32_t width, uint32_t height,
                                 ImageFormat imageFormat,
                                 TDLDataType pix_data_type,
                                 VIDEO_FRAME_INFO_S* frame) {
  PIXEL_FORMAT_E pixel_format = convertPixelFormat(imageFormat, pix_data_type);
  if (pixel_format == PIXEL_FORMAT_MAX) {
    LOGE("convertPixelFormat failed, imageFormat: %d,pix_data_type:%d",
         (int)imageFormat, (int)pix_data_type);
    return -1;
  }

  VB_CAL_CONFIG_S stVbConf;

  COMMON_GetPicBufferConfig(width, height, pixel_format, DATA_BITWIDTH_8,
                            COMPRESS_MODE_NONE, DEFAULT_ALIGN, &stVbConf);

  if (stVbConf.plane_num == 0) {
    LOGE("not supported format %u", pixel_format);
    return -1;
  }
  memset(frame, 0, sizeof(VIDEO_FRAME_INFO_S));

  VIDEO_FRAME_S* vFrame = &frame->stVFrame;

  vFrame->enCompressMode = COMPRESS_MODE_NONE;
  vFrame->enPixelFormat = pixel_format;
  vFrame->enVideoFormat = VIDEO_FORMAT_LINEAR;
  vFrame->enColorGamut = COLOR_GAMUT_BT709;
  vFrame->u32TimeRef = 0;
  vFrame->u64PTS = 0;
  vFrame->enDynamicRange = DYNAMIC_RANGE_SDR8;

  vFrame->u32Width = width;
  vFrame->u32Height = height;
  vFrame->u32Stride[0] = stVbConf.u32MainStride;
  vFrame->u32Stride[1] = stVbConf.u32CStride;
  vFrame->u32Stride[2] = stVbConf.u32CStride;

  vFrame->u32Length[0] = stVbConf.u32MainYSize;
  vFrame->u32Length[1] = stVbConf.u32MainCSize;
  if (stVbConf.plane_num == 3) vFrame->u32Length[2] = stVbConf.u32MainCSize;

  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  return 0;
}

int32_t VPSSImage::setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                               uint32_t length) {
  if (frame_.stVFrame.u64PhyAddr[0] != 0 ||
      frame_.stVFrame.pu8VirAddr[0] != nullptr) {
    LOGE("setupMemory failed, frame already initialized\n");
    return -1;
  }

  if (getImageByteSize() != length) {
    LOGE("setupMemory failed, length not match, expected: %d, actual: %d",
         getImageByteSize(), length);
    return -1;
  }
  frame_.stVFrame.u64PhyAddr[0] = phy_addr;
  frame_.stVFrame.pu8VirAddr[0] = vir_addr;
  for (int i = 1; i < 3; i++) {
    if (frame_.stVFrame.u32Length[i] != 0) {
      frame_.stVFrame.u64PhyAddr[i] =
          frame_.stVFrame.u64PhyAddr[i - 1] + frame_.stVFrame.u32Length[i - 1];
      frame_.stVFrame.pu8VirAddr[i] =
          frame_.stVFrame.pu8VirAddr[i - 1] + frame_.stVFrame.u32Length[i - 1];
    } else {
      LOGI("plane %d is not used", i);
      break;
    }
  }

  LOGI("setupMemory done,width:%d,height:%d,format:%d,addr:%p,phyaddr:%#llx",
       frame_.stVFrame.u32Width, frame_.stVFrame.u32Height,
       frame_.stVFrame.enPixelFormat, frame_.stVFrame.pu8VirAddr[0],
       frame_.stVFrame.u64PhyAddr[0]);
  return 0;
}

PIXEL_FORMAT_E VPSSImage::convertPixelFormat(ImageFormat img_format,
                                             TDLDataType pix_data_type) {
  PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;

  if (img_format == ImageFormat::GRAY) {
    pixel_format = PIXEL_FORMAT_YUV_400;
  } else if (img_format == ImageFormat::YUV420SP_UV) {
    pixel_format = PIXEL_FORMAT_NV12;
  } else if (img_format == ImageFormat::YUV420SP_VU) {
    pixel_format = PIXEL_FORMAT_NV21;
  } else if (img_format == ImageFormat::YUV420P_UV) {
    LOGE("YUV420SP_VU not support, imageFormat: %d", (int)img_format);
  } else if (img_format == ImageFormat::YUV420P_VU) {
    LOGE("YUV420SP_VU not support, imageFormat: %d", (int)img_format);
  } else if (img_format == ImageFormat::RGB_PACKED) {
    pixel_format = PIXEL_FORMAT_RGB_888;
  } else if (img_format == ImageFormat::BGR_PACKED) {
    pixel_format = PIXEL_FORMAT_BGR_888;
  } else if (img_format == ImageFormat::RGB_PLANAR) {
    pixel_format = PIXEL_FORMAT_RGB_888_PLANAR;
  } else if (img_format == ImageFormat::BGR_PLANAR) {
    pixel_format = PIXEL_FORMAT_BGR_888_PLANAR;
  } else {
    LOGE("imageFormat not support, imageFormat: %d", (int)img_format);
    pixel_format = PIXEL_FORMAT_MAX;
  }

  if (pix_data_type == TDLDataType::UINT8 &&
      (img_format == ImageFormat::RGB_PLANAR ||
       img_format == ImageFormat::BGR_PLANAR)) {
    LOGW("special case, imageFormat: %d,pix_data_type: %d", (int)img_format,
         (int)pix_data_type);
    pixel_format = PIXEL_FORMAT_UINT8_C3_PLANAR;
  }

  if (pix_data_type != TDLDataType::INT8 &&
      pix_data_type != TDLDataType::UINT8) {
    LOGW("pix_data_type not support, pix_data_type: %d", (int)pix_data_type);
    pixel_format = PIXEL_FORMAT_MAX;
  }

  return pixel_format;
}

uint32_t VPSSImage::getPlaneNum() const {
  if (frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_RGB_888_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_BGR_888_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_422 ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_420 ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_PLANAR_444 ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_HSV_888_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_FP32_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_INT32_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_UINT32_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_BF16_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_INT16_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_UINT16_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_INT8_C3_PLANAR ||
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_UINT8_C3_PLANAR) {
    return 3;
  } else if (frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_NV12 ||
             frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_NV21) {
    return 2;
  } else if (frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_YUV_400 ||
             frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_BGR_888 ||
             frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_RGB_888) {
    return 1;
  } else {
    LOGE("not supported format %u", frame_.stVFrame.enPixelFormat);
    return -1;
  }
}

std::vector<uint32_t> VPSSImage::getStrides() const {
  LOGI("getStrides,stride0:%d,stride1:%d,stride2:%d",
       frame_.stVFrame.u32Stride[0], frame_.stVFrame.u32Stride[1],
       frame_.stVFrame.u32Stride[2]);
  return {frame_.stVFrame.u32Stride[0], frame_.stVFrame.u32Stride[1],
          frame_.stVFrame.u32Stride[2]};
}

std::vector<uint64_t> VPSSImage::getPhysicalAddress() const {
  std::vector<uint64_t> physical_address;
  for (size_t i = 0; i < getPlaneNum(); i++) {
    physical_address.push_back(frame_.stVFrame.u64PhyAddr[i]);
  }
  return physical_address;
}

std::vector<uint8_t*> VPSSImage::getVirtualAddress() const {
  std::vector<uint8_t*> virtual_address;
  for (size_t i = 0; i < getPlaneNum(); i++) {
    virtual_address.push_back(frame_.stVFrame.pu8VirAddr[i]);
  }
  return virtual_address;
}

uint32_t VPSSImage::getWidth() const { return frame_.stVFrame.u32Width; }

uint32_t VPSSImage::getHeight() const { return frame_.stVFrame.u32Height; }

uint32_t VPSSImage::getInternalType() const {
  return (uint32_t)frame_.stVFrame.enPixelFormat;
}
void* VPSSImage::getInternalData() const { return (void*)&frame_; }

uint32_t VPSSImage::getImageByteSize() const {
  uint32_t size = 0;
  for (size_t i = 0; i < getPlaneNum(); i++) {
    size += frame_.stVFrame.u32Length[i];
  }
  return size;
}

VIDEO_FRAME_INFO_S* VPSSImage::getFrame() const {
  return const_cast<VIDEO_FRAME_INFO_S*>(&frame_);
}

PIXEL_FORMAT_E VPSSImage::getPixelFormat() const {
  return frame_.stVFrame.enPixelFormat;
}

uint32_t VPSSImage::getVbPoolId() const {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr");
    return UINT32_MAX;
  }
  return memory_block_->id;
}

int32_t VPSSImage::extractImageInfo(const VIDEO_FRAME_INFO_S& frame) {
  PIXEL_FORMAT_E pixel_format = frame.stVFrame.enPixelFormat;
  // uint32_t width = frame.stVFrame.u32Width;
  // uint32_t height = frame.stVFrame.u32Height;

  image_type_ = ImageType::VPSS_FRAME;

  if (pixel_format == PIXEL_FORMAT_YUV_400) {
    image_format_ = ImageFormat::GRAY;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_NV12) {
    image_format_ = ImageFormat::YUV420SP_UV;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_NV21) {
    image_format_ = ImageFormat::YUV420SP_VU;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_RGB_888) {
    image_format_ = ImageFormat::RGB_PACKED;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_BGR_888) {
    image_format_ = ImageFormat::BGR_PACKED;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_RGB_888_PLANAR) {
    image_format_ = ImageFormat::RGB_PLANAR;
    pix_data_type_ = TDLDataType::UINT8;
  } else if (pixel_format == PIXEL_FORMAT_BGR_888_PLANAR) {
    image_format_ = ImageFormat::BGR_PLANAR;
    pix_data_type_ = TDLDataType::UINT8;
  } else {
    LOGE("pixel_format not supported, pixel_format: %d", pixel_format);
    return -1;
  }
  memory_block_ = std::make_unique<MemoryBlock>();
  memory_block_->id = 0;
  memory_block_->size = getImageByteSize();
  memory_block_->physicalAddress = frame.stVFrame.u64PhyAddr[0];
  memory_block_->virtualAddress = frame.stVFrame.pu8VirAddr[0];
  memory_block_->own_memory = false;

  return 0;
}

int32_t VPSSImage::restoreVirtualAddress(bool check_swap_rgb) {
  if (memory_block_ == nullptr) {
    LOGW("memory_block_ is nullptr");
    return -1;
  }
  if (memory_block_->virtualAddress == nullptr) {
    LOGW("virtualAddress is nullptr");
    return -1;
  }
  if (frame_.stVFrame.pu8VirAddr[0] != 0) {
    LOGW("virtualAddress is not nullptr,do not restore");
    return -1;
  }
  int num_plane_restored = 1;
  frame_.stVFrame.pu8VirAddr[0] = (uint8_t*)(memory_block_->virtualAddress);

  for (int i = 1; i < 3; i++) {
    if (frame_.stVFrame.u32Length[i] != 0) {
      frame_.stVFrame.pu8VirAddr[i] =
          frame_.stVFrame.pu8VirAddr[i - 1] + frame_.stVFrame.u32Length[i - 1];
      num_plane_restored++;
    } else {
      break;
    }
  }
  if (check_swap_rgb &&
      frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_UINT8_C3_PLANAR &&
      image_format_ == ImageFormat::BGR_PLANAR) {
    uint8_t* ptr_r = frame_.stVFrame.pu8VirAddr[0];
    frame_.stVFrame.pu8VirAddr[0] = frame_.stVFrame.pu8VirAddr[2];
    frame_.stVFrame.pu8VirAddr[2] = ptr_r;
    LOGI("swap r and b,width:%d,height:%d,format:%d,addr:%lx,phyaddr:%#llx",
         frame_.stVFrame.u32Width, frame_.stVFrame.u32Height,
         frame_.stVFrame.enPixelFormat, frame_.stVFrame.pu8VirAddr[0],
         frame_.stVFrame.u64PhyAddr[0]);
  }

  LOGI("restoreVirtualAddress done,num_plane_restored:%d", num_plane_restored);
  return 0;
}

int32_t VPSSImage::checkToSwapRGB() {
  if (frame_.stVFrame.enPixelFormat == PIXEL_FORMAT_UINT8_C3_PLANAR &&
      image_format_ == ImageFormat::BGR_PLANAR) {
    // PIXEL_FORMAT_UINT8_C3_PLANAR is rgb order,need to swap r and b
    frame_.stVFrame.pu8VirAddr[0] = (uint8_t*)(memory_block_->virtualAddress) +
                                    frame_.stVFrame.u32Length[0] +
                                    frame_.stVFrame.u32Length[1];
    frame_.stVFrame.pu8VirAddr[2] = (uint8_t*)(memory_block_->virtualAddress);
    frame_.stVFrame.u64PhyAddr[0] = memory_block_->physicalAddress +
                                    frame_.stVFrame.u32Length[0] +
                                    frame_.stVFrame.u32Length[1];
    frame_.stVFrame.u64PhyAddr[2] = memory_block_->physicalAddress;

    LOGI("swap r and b,width:%d,height:%d,format:%d,addr:%lx,phyaddr:%#llx",
         frame_.stVFrame.u32Width, frame_.stVFrame.u32Height,
         frame_.stVFrame.enPixelFormat, frame_.stVFrame.pu8VirAddr[0],
         frame_.stVFrame.u64PhyAddr[0]);
  }
  return 0;
}
