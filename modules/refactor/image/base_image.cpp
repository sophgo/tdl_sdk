#include "image/base_image.hpp"

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <random>

#include "cvi_tdl_log.hpp"
#include "utils/common_utils.hpp"
BaseImage::BaseImage() {
#ifdef __SOPHON__
  memory_pool_ =
      BaseMemoryPoolFactory::createMemoryPool(MemoryPoolType::BM_SOC_DEVICE);
#else
  memory_pool_ =
      BaseMemoryPoolFactory::createMemoryPool(MemoryPoolType::CVI_SOC_DEVICE);
#endif
}

int32_t BaseImage::randomFill() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  auto virtual_address = getVirtualAddress();
  uint32_t image_size = getImageByteSize();
  if (virtual_address.empty() || image_size == 0) {
    return -1;
  }
  uint8_t* data = virtual_address[0];
  for (size_t i = 0; i < image_size; ++i) {
    data[i] = static_cast<uint8_t>(dis(gen));
  }
  int32_t ret = flushCache();
  if (ret != 0) {
    std::cerr << "base image flush cache failed" << std::endl;
    return ret;
  }
  return 0;
}

bool BaseImage::isPlanar() const {
  auto image_format = getImageFormat();
  if (image_format == ImageFormat::RGB_PLANAR ||
      image_format == ImageFormat::BGR_PLANAR ||
      image_format == ImageFormat::GRAY) {
    return true;
  }
  return false;
}
int32_t BaseImage::allocateMemory() {
  uint32_t image_size = getImageByteSize();
  if (image_size == 0) {
    LOGE("image size is 0");
    return -1;
  }
  if (memory_pool_ == nullptr) {
    LOGE("memory pool is nullptr");
    return -1;
  }
  if (memory_block_ != nullptr) {
    LOGE("memory block is not nullptr");
    return -1;
  }
  memory_block_ = memory_pool_->allocate(image_size);
  if (memory_block_ == nullptr) {
    LOGE("allocate memory failed");
    return -1;
  }
  int32_t ret = setupMemory(
      memory_block_->physicalAddress,
      static_cast<uint8_t*>(memory_block_->virtualAddress), image_size);
  if (ret != 0) {
    LOGE("setup memory failed");
    return ret;
  }
  return 0;
}

int32_t BaseImage::freeMemory() {
  if (memory_block_ == nullptr) {
    LOGE("memory block is nullptr");
    return -1;
  }
  if (memory_block_->own_memory) {
    if (is_local_mempool_) {
      memory_pool_->release(memory_block_);
    } else {
      memory_pool_->recycle(memory_block_);
    }
  }

  memory_block_ = nullptr;

  return 0;
}

bool BaseImage::isInitialized() const { return memory_block_ != nullptr; }
bool BaseImage::isAligned() const {
  int row_bytes = getWidth() * get_data_type_size(getPixDataType());
  bool is_aligned = row_bytes == getStrides()[0];
  return is_aligned;
}
int32_t BaseImage::invalidateCache() {
  int32_t ret = memory_pool_->invalidateCache(memory_block_);
  return ret;
}

int32_t BaseImage::flushCache() {
  int32_t ret = memory_pool_->flushCache(memory_block_);
  return ret;
}
int32_t BaseImage::setupMemoryBlock(
    std::unique_ptr<MemoryBlock>& memory_block) {
  int32_t ret =
      setupMemory(memory_block->physicalAddress,
                  (uint8_t*)memory_block->virtualAddress, memory_block->size);
  if (ret != 0) {
    LOGE("setup memory failed");
    return ret;
  }
  memory_block_ = std::move(memory_block);
  return 0;
}

int32_t BaseImage::readImage(const std::string& file_path) {
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
  auto vir_addr = getVirtualAddress();
  std::vector<uint32_t> strides = getStrides();
  for (int r = 0; r < img.rows; r++) {
    uint8_t* ptr = img.data + r * img.step[0];
    uint8_t* dst = (uint8_t*)vir_addr[0] + r * strides[0];
    memcpy(dst, ptr, img.cols * 3);
  }
  ret = flushCache();
  if (ret != 0) {
    LOGE("flushCache failed, ret: %d", ret);
    return -1;
  }
  return 0;
}

int32_t BaseImage::writeImage(const std::string& file_path) {
  // Implementation here
  if (image_format_ != ImageFormat::BGR_PACKED &&
      image_format_ != ImageFormat::RGB_PACKED &&
      image_format_ != ImageFormat::GRAY &&
      image_format_ != ImageFormat::BGR_PLANAR &&
      image_format_ != ImageFormat::RGB_PLANAR) {
    LOGE("writeImage failed, image format not supported");
    return -1;
  }
  cv::Mat img(getHeight(), getWidth(), CV_8UC3);
  int32_t ret = invalidateCache();
  if (ret != 0) {
    LOGE("invalidateCache failed, ret: %d", ret);
    return -1;
  }
  auto vir_addr = getVirtualAddress();
  std::vector<uint32_t> strides = getStrides();
  for (int r = 0; r < img.rows; r++) {
    if (image_format_ == ImageFormat::BGR_PACKED ||
        image_format_ == ImageFormat::RGB_PACKED) {
      uint8_t* src = vir_addr[0] + r * strides[0];
      uint8_t* dst = img.data + r * img.step[0];
      memcpy(dst, src, img.cols * 3);
    } else if (image_format_ == ImageFormat::BGR_PLANAR ||
               image_format_ == ImageFormat::RGB_PLANAR) {
      uint8_t* dst = img.data + r * img.step[0];
      uint8_t* src1 = (uint8_t*)vir_addr[0] + r * strides[0];
      uint8_t* src2 = (uint8_t*)vir_addr[1] + r * strides[1];
      uint8_t* src3 = (uint8_t*)vir_addr[2] + r * strides[2];
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