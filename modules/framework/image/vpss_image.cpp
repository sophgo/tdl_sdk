#include "image/vpss_image.hpp"

#include "cvi_tdl_log.hpp"
#include "cvi_vb.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#define SCALAR_4096_ALIGN_BUG 0x1000
VPSSImage::VPSSImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
                     ImagePixDataType pix_data_type,
                     std::unique_ptr<BaseMemoryPool> memory_pool) {
  int32_t ret =
      initFrameInfo(width, height, imageFormat, pix_data_type, &frame_);
  if (ret != 0) {
    throw std::runtime_error("initFrameInfo failed");
  }

  VIDEO_FRAME_S* vFrame = &frame_.stVFrame;
  CVI_U32 u32MapSize =
      vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];

  uint64_t phy_addr = 0;
  void* vir_addr = nullptr;
  // if (memory_pool == nullptr) {
  //   int ret = CVI_SYS_IonAlloc(&phy_addr, &vir_addr, "vpss_image",
  //   u32MapSize); if (ret != CVI_SUCCESS) {
  //     throw std::runtime_error("CVI_SYS_IonAlloc failed");
  //   }
  //   is_from_pool_ = false;
  // } else {
  //   if (memory_block->size < u32MapSize) {
  //     throw std::runtime_error("memory_block size is too small");
  //   }
  //   phy_addr = memory_block->physicalAddress;
  //   vir_addr = memory_block->virtualAddress;
  //   memory_block_ = std::move(memory_block);
  //   is_from_pool_ = true;
  // }

  ret = setupMemory(phy_addr, (uint8_t*)vir_addr, u32MapSize);
  if (ret != CVI_SUCCESS) {
    LOGE("setupMemory failed, ret: %d", ret);
    throw std::runtime_error("setupMemory failed");
  }
  LOGI(
      "VPSSImage init "
      "done,pyaddr:%lx,viraddr:%lx,widht:%d,height:%d,format:%d,pix_data_type:%"
      "d",
      vFrame->u64PhyAddr[0], vFrame->pu8VirAddr[0], vFrame->u32Width,
      vFrame->u32Height, (int)imageFormat, (int)pix_data_type);
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
}

VPSSImage::VPSSImage(const VIDEO_FRAME_INFO_S& frame) {
  frame_ = frame;
  is_from_pool_ = false;
  memory_block_ = nullptr;
}

VPSSImage::VPSSImage() {
  is_from_pool_ = false;
  memory_block_ = nullptr;
  memset(&frame_, 0, sizeof(VIDEO_FRAME_INFO_S));
}

VPSSImage::~VPSSImage() {
  CVI_S32 ret = CVI_SUCCESS;
  if (memory_block_ != nullptr && memory_block_->own_memory) {
    if (memory_block_->id != UINT32_MAX) {
      CVI_VB_DestroyPool(memory_block_->id);
    } else {
      ret = CVI_SYS_IonFree(memory_block_->physicalAddress,
                            (void*)memory_block_->virtualAddress);
    }
    memory_block_ = nullptr;
  }
  if (ret != CVI_SUCCESS) {
    LOGE(
        "VPSSImage::~VPSSImage "
        "failed,ret:%d,is_from_pool_:%d,phyaddr:%lx,viraddr:%lx,width:%d,"
        "height:%d,format:%d,pix_data_type:%d",
        ret, is_from_pool_, frame_.stVFrame.u64PhyAddr[0],
        frame_.stVFrame.pu8VirAddr[0], frame_.stVFrame.u32Width,
        frame_.stVFrame.u32Height, (int)image_format_, (int)pix_data_type_);
  } else {
    LOGI(
        "VPSSImage::~VPSSImage "
        "done,ret:%d,is_from_pool_:%d,phyaddr:%lx,viraddr:%lx,width:%d,"
        "height:%d,format:%d,pix_data_type:%d",
        ret, is_from_pool_, frame_.stVFrame.u64PhyAddr[0],
        frame_.stVFrame.pu8VirAddr[0], frame_.stVFrame.u32Width,
        frame_.stVFrame.u32Height, (int)image_format_, (int)pix_data_type_);
  }
  is_from_pool_ = false;
  memory_block_ = nullptr;
  memset(&frame_, 0, sizeof(VIDEO_FRAME_INFO_S));
}

int32_t VPSSImage::prepareImageInfo(uint32_t width, uint32_t height,
                                    ImageFormat imageFormat,
                                    ImagePixDataType pix_data_type) {
  return initFrameInfo(width, height, imageFormat, pix_data_type, &frame_);
}
int32_t VPSSImage::allocateMemory() {
  CVI_U32 u32MapSize = getImageByteSize();

  uint64_t phy_addr = 0;
  void* vir_addr = nullptr;
  int32_t ret =
      CVI_SYS_IonAlloc(&phy_addr, &vir_addr, "vpss_image", u32MapSize);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_SYS_IonAlloc failed, ret: %d", ret);
    return -1;
  }

  ret = setupMemory(phy_addr, (uint8_t*)vir_addr, u32MapSize);
  if (ret != CVI_SUCCESS) {
    LOGE("setupMemory failed, ret: %d", ret);
    return -1;
  }
  memory_block_ = std::make_unique<MemoryBlock>();
  memory_block_->physicalAddress = phy_addr;
  memory_block_->virtualAddress = vir_addr;
  memory_block_->size = u32MapSize;
  memory_block_->own_memory = true;

  return ret;
}
int32_t VPSSImage::setupMemoryBlock(
    std::unique_ptr<MemoryBlock>& memory_block) {
  if (memory_block == nullptr) {
    return -1;
  }

  int32_t ret =
      setupMemory(memory_block->physicalAddress,
                  (uint8_t*)memory_block->virtualAddress, memory_block->size);
  if (ret != 0) {
    LOGE("setup memory failed");
    return ret;
  }
  memory_block_ = std::move(memory_block);
  is_from_pool_ = true;

  return 0;
}

bool VPSSImage::isInitialized() const {
  bool is_initialized = frame_.stVFrame.u64PhyAddr[0] != 0 &&
                        frame_.stVFrame.pu8VirAddr[0] != nullptr &&
                        frame_.stVFrame.u32Width != 0 &&
                        frame_.stVFrame.u32Height != 0;
  return is_initialized;
}

int32_t VPSSImage::initFrameInfo(uint32_t width, uint32_t height,
                                 ImageFormat imageFormat,
                                 ImagePixDataType pix_data_type,
                                 VIDEO_FRAME_INFO_S* frame) {
  PIXEL_FORMAT_E pixel_format = convertPixelFormat(imageFormat, pix_data_type);
  if (pixel_format == PIXEL_FORMAT_MAX) {
    LOGE("convertPixelFormat failed, imageFormat: %d", (int)imageFormat);
    return -1;
  }

  VIDEO_FRAME_S* vFrame = &frame->stVFrame;
  memset(vFrame, 0, sizeof(VIDEO_FRAME_S));
  vFrame->enCompressMode = COMPRESS_MODE_NONE;
  vFrame->enPixelFormat = pixel_format;
  vFrame->enVideoFormat = VIDEO_FORMAT_LINEAR;
  vFrame->enColorGamut = COLOR_GAMUT_BT709;
  vFrame->u32TimeRef = 0;
  vFrame->u64PTS = 0;
  vFrame->enDynamicRange = DYNAMIC_RANGE_SDR8;

  vFrame->u32Width = width;
  vFrame->u32Height = height;
  switch (vFrame->enPixelFormat) {
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width * 3, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
      break;
    }

    case PIXEL_FORMAT_RGB_888_PLANAR:
    case PIXEL_FORMAT_BGR_888_PLANAR: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = vFrame->u32Stride[0];
      vFrame->u32Stride[2] = vFrame->u32Stride[0];
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = vFrame->u32Length[0];
      vFrame->u32Length[2] = vFrame->u32Length[0];
      break;
    }

    case PIXEL_FORMAT_YUV_PLANAR_422: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width >> 1, DEFAULT_ALIGN);
      vFrame->u32Stride[2] = ALIGN(vFrame->u32Width >> 1, DEFAULT_ALIGN);
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = ALIGN(vFrame->u32Stride[1] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] = ALIGN(vFrame->u32Stride[2] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      break;
    }

    case PIXEL_FORMAT_YUV_PLANAR_420: {
      uint32_t newHeight = ALIGN(vFrame->u32Height, 2);
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width >> 1, DEFAULT_ALIGN);
      vFrame->u32Stride[2] = ALIGN(vFrame->u32Width >> 1, DEFAULT_ALIGN);
      vFrame->u32Length[0] =
          ALIGN(vFrame->u32Stride[0] * newHeight, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] =
          ALIGN(vFrame->u32Stride[1] * newHeight / 2, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] =
          ALIGN(vFrame->u32Stride[2] * newHeight / 2, SCALAR_4096_ALIGN_BUG);
      break;
    }

    case PIXEL_FORMAT_YUV_400: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
      break;
    }

    case PIXEL_FORMAT_NV12: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height / 2,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] = 0;
      break;
    }

    case PIXEL_FORMAT_NV21: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width, DEFAULT_ALIGN);
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height / 2,
                                   SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] = 0;
      break;
    }

    case PIXEL_FORMAT_FP32_C1: {
      vFrame->u32Stride[0] =
          ALIGN(vFrame->u32Width, DEFAULT_ALIGN) * sizeof(float);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
      break;
    }

    case PIXEL_FORMAT_BF16_C1: {
      vFrame->u32Stride[0] =
          ALIGN(vFrame->u32Width, DEFAULT_ALIGN) * sizeof(uint16_t);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
      break;
    }

    default:
      LOGE("Currently unsupported format %u\n", vFrame->enPixelFormat);
      return -1;
  }
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
  frame_.stVFrame.u64PhyAddr[1] =
      frame_.stVFrame.u64PhyAddr[0] + frame_.stVFrame.u32Length[0];
  frame_.stVFrame.u64PhyAddr[2] =
      frame_.stVFrame.u64PhyAddr[1] + frame_.stVFrame.u32Length[1];
  frame_.stVFrame.pu8VirAddr[1] =
      frame_.stVFrame.pu8VirAddr[0] + frame_.stVFrame.u32Length[0];
  frame_.stVFrame.pu8VirAddr[2] =
      frame_.stVFrame.pu8VirAddr[1] + frame_.stVFrame.u32Length[1];

  return 0;
}

PIXEL_FORMAT_E VPSSImage::convertPixelFormat(
    ImageFormat imageFormat, ImagePixDataType pix_data_type) const {
  PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;

  if (imageFormat == ImageFormat::GRAY) {
    pixel_format = PIXEL_FORMAT_YUV_400;
  } else if (imageFormat == ImageFormat::YUV420SP_UV) {
    pixel_format = PIXEL_FORMAT_NV12;
  } else if (imageFormat == ImageFormat::YUV420SP_VU) {
    LOGE("YUV420SP_VU not support, imageFormat: %d", (int)imageFormat);
  } else if (imageFormat == ImageFormat::YUV420P_UV) {
    pixel_format = PIXEL_FORMAT_NV21;
  } else if (imageFormat == ImageFormat::YUV420P_VU) {
    LOGE("YUV420SP_VU not support, imageFormat: %d", (int)imageFormat);
  } else if (imageFormat == ImageFormat::RGB_PACKED) {
    pixel_format = PIXEL_FORMAT_RGB_888;
  } else if (imageFormat == ImageFormat::BGR_PACKED) {
    pixel_format = PIXEL_FORMAT_BGR_888;
  } else if (imageFormat == ImageFormat::RGB_PLANAR) {
    pixel_format = PIXEL_FORMAT_RGB_888_PLANAR;
  } else if (imageFormat == ImageFormat::BGR_PLANAR) {
    pixel_format = PIXEL_FORMAT_BGR_888_PLANAR;
  } else {
    LOGE("imageFormat not support, imageFormat: %d", (int)imageFormat);
    pixel_format = PIXEL_FORMAT_MAX;
  }

  if (pix_data_type != ImagePixDataType::INT8 &&
      pix_data_type != ImagePixDataType::UINT8) {
    LOGE("pix_data_type not support, pix_data_type: %d", (int)pix_data_type);
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
  } else {
    return 1;
  }
}

std::vector<uint32_t> VPSSImage::getStrides() const {
  return {frame_.stVFrame.u32Stride[0], frame_.stVFrame.u32Stride[1],
          frame_.stVFrame.u32Stride[2]};
}

int32_t VPSSImage::invalidateCache() {
  uint32_t image_size = frame_.stVFrame.u32Length[0] +
                        frame_.stVFrame.u32Length[1] +
                        frame_.stVFrame.u32Length[2];
  CVI_S32 ret = CVI_SYS_IonInvalidateCache(frame_.stVFrame.u64PhyAddr[0],
                                           (void*)frame_.stVFrame.pu8VirAddr[0],
                                           image_size);
  LOGI(
      "invalidateCache "
      "done,ret:%d,phyaddr:%lx,viraddr:%lx,width:%d,height:%d,format:%d,pix_"
      "data_type:%d",
      ret, frame_.stVFrame.u64PhyAddr[0], frame_.stVFrame.pu8VirAddr[0],
      frame_.stVFrame.u32Width, frame_.stVFrame.u32Height, (int)image_format_,
      (int)pix_data_type_);
  return (int32_t)ret;
}

int32_t VPSSImage::flushCache() {
  uint32_t image_size = frame_.stVFrame.u32Length[0] +
                        frame_.stVFrame.u32Length[1] +
                        frame_.stVFrame.u32Length[2];
  CVI_S32 ret =
      CVI_SYS_IonFlushCache(frame_.stVFrame.u64PhyAddr[0],
                            (void*)frame_.stVFrame.pu8VirAddr[0], image_size);
  LOGI(
      "flushCache "
      "done,ret:%d,phyaddr:%lx,viraddr:%lx,width:%d,height:%d,format:%d,pix_"
      "data_type:%d",
      ret, frame_.stVFrame.u64PhyAddr[0], frame_.stVFrame.pu8VirAddr[0],
      frame_.stVFrame.u32Width, frame_.stVFrame.u32Height, (int)image_format_,
      (int)pix_data_type_);
  return (int32_t)ret;
}

std::vector<uint64_t> VPSSImage::getPhysicalAddress() const {
  std::vector<uint64_t> physical_address;
  for (int i = 0; i < getPlaneNum(); i++) {
    physical_address.push_back(frame_.stVFrame.u64PhyAddr[i]);
  }
  return physical_address;
}

std::vector<uint8_t*> VPSSImage::getVirtualAddress() const {
  std::vector<uint8_t*> virtual_address;
  for (int i = 0; i < getPlaneNum(); i++) {
    virtual_address.push_back(frame_.stVFrame.pu8VirAddr[i]);
  }
  return virtual_address;
}

uint32_t VPSSImage::getWidth() const { return frame_.stVFrame.u32Width; }

uint32_t VPSSImage::getHeight() const { return frame_.stVFrame.u32Height; }

uint32_t VPSSImage::getInternalType() {
  return (uint32_t)frame_.stVFrame.enPixelFormat;
}
void* VPSSImage::getInternalData() const { return (void*)&frame_; }

uint32_t VPSSImage::getImageByteSize() const {
  uint32_t size = 0;
  for (int i = 0; i < getPlaneNum(); i++) {
    size += frame_.stVFrame.u32Length[i];
  }
  return size;
}

std::string VPSSImage::getDeviceType() const { return "VPSS"; }

VIDEO_FRAME_INFO_S* VPSSImage::getFrame() const {
  return const_cast<VIDEO_FRAME_INFO_S*>(&frame_);
}

void VPSSImage::setFrame(const VIDEO_FRAME_INFO_S& frame) { frame_ = frame; }

PIXEL_FORMAT_E VPSSImage::getPixelFormat() const {
  return frame_.stVFrame.enPixelFormat;
}

int32_t VPSSImage::readImage(const std::string& file_path) {
  // Implementation here
  cv::Mat img = cv::imread(file_path);
  if (img.empty()) {
    LOGE("Failed to load image from file: %s", file_path.c_str());
    return -1;
  } else {
    LOGI("read image %s done, width: %d, height: %d", file_path.c_str(),
         img.cols, img.rows);
  }
  int32_t ret = prepareImageInfo(img.cols, img.rows, ImageFormat::BGR_PACKED,
                                 ImagePixDataType::UINT8);
  if (ret != 0) {
    LOGE("prepareImageInfo failed, ret: %d", ret);
    return -1;
  }
  ret = allocateMemory();
  if (ret != 0) {
    LOGE("allocateMemory failed, ret: %d", ret);
    return -1;
  }
  for (int r = 0; r < img.rows; r++) {
    uint8_t* ptr = img.data + r * img.step[0];
    uint8_t* dst = (uint8_t*)frame_.stVFrame.pu8VirAddr[0] +
                   r * frame_.stVFrame.u32Stride[0];
    memcpy(dst, ptr, img.cols * 3);
  }
  ret = flushCache();
  if (ret != CVI_SUCCESS) {
    LOGE("flushCache failed, ret: %d", ret);
    return -1;
  }
  return 0;
}

int32_t VPSSImage::writeImage(const std::string& file_path) {
  // Implementation here
  if (image_format_ != ImageFormat::BGR_PACKED &&
      image_format_ != ImageFormat::RGB_PACKED &&
      image_format_ != ImageFormat::GRAY &&
      image_format_ != ImageFormat::BGR_PLANAR &&
      image_format_ != ImageFormat::RGB_PLANAR) {
    LOGE("writeImage failed, image format not supported");
    return -1;
  }

  cv::Mat img(frame_.stVFrame.u32Height, frame_.stVFrame.u32Width, CV_8UC3);
  int32_t ret = invalidateCache();
  if (ret != CVI_SUCCESS) {
    LOGE("invalidateCache failed, ret: %d", ret);
    return -1;
  }
  auto vir_addr = getVirtualAddress();
  for (int r = 0; r < img.rows; r++) {
    if (image_format_ == ImageFormat::BGR_PACKED ||
        image_format_ == ImageFormat::RGB_PACKED) {
      uint8_t* src = vir_addr[0] + r * frame_.stVFrame.u32Stride[0];
      uint8_t* dst = img.data + r * img.step[0];
      memcpy(dst, src, img.cols * 3);
    } else if (image_format_ == ImageFormat::BGR_PLANAR ||
               image_format_ == ImageFormat::RGB_PLANAR) {
      uint8_t* dst = img.data + r * img.step[0];
      uint8_t* src1 = (uint8_t*)vir_addr[0] + r * frame_.stVFrame.u32Stride[0];
      uint8_t* src2 = (uint8_t*)vir_addr[1] + r * frame_.stVFrame.u32Stride[1];
      uint8_t* src3 = (uint8_t*)vir_addr[2] + r * frame_.stVFrame.u32Stride[2];
      for (int c = 0; c < img.cols; c++) {
        dst[c * 3] = src1[c];
        dst[c * 3 + 1] = src2[c];
        dst[c * 3 + 2] = src3[c];
      }
    }
  }
  cv::imwrite(file_path, img);
  return 0;
}

uint32_t VPSSImage::getVbPoolId() const {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr");
    return UINT32_MAX;
  }
  return memory_block_->id;
}
