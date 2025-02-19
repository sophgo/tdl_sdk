#include "image/base_image.hpp"

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <random>

#include "cvi_tdl_log.hpp"
#include "utils/common_utils.hpp"

BaseImage::BaseImage() : memory_pool_(nullptr), memory_block_(nullptr) {}
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
  LOGI(
      "allocateMemory "
      "done,width:%d,height:%d,format:%d,pix_type:%d,virtual_address:%lx,"
      "physical_address:%lx",
      getWidth(), getHeight(), getImageFormat(), getPixDataType(),
      memory_block_->virtualAddress, memory_block_->physicalAddress);
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

int32_t BaseImage::setMemoryPool(std::shared_ptr<BaseMemoryPool> memory_pool) {
  memory_pool_ = memory_pool;
  return 0;
}
