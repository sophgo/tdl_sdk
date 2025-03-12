#include "image/base_image.hpp"

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <random>

#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

BaseImage::BaseImage(ImageType image_type)
    : memory_pool_(nullptr), memory_block_(nullptr), image_type_(image_type) {}

BaseImage::~BaseImage() {
  if (memory_block_ != nullptr && memory_block_->own_memory) {
    LOGI("base image release memory block,size:%d,phy_addr:%p,virtual_addr:%p",
         memory_block_->size, (void*)memory_block_->physicalAddress,
         (void*)memory_block_->virtualAddress);
    if (memory_pool_ != nullptr) {
      memory_pool_->release(memory_block_);
    } else {
      LOGE("memory pool is nullptr, memory block is not nullptr");
      assert(false);
    }
    memory_block_ = nullptr;
  }
}

int32_t BaseImage::prepareImageInfo(uint32_t width, uint32_t height,
                                    ImageFormat imageFormat,
                                    TDLDataType pixDataType,
                                    uint32_t align_size) {
  width_ = width;
  height_ = height;
  image_format_ = imageFormat;
  pix_data_type_ = pixDataType;
  align_size_ = align_size;
  return initImageInfo();
}

int32_t BaseImage::initImageInfo() {
  uint32_t pix_size = get_data_type_size(getPixDataType());
  if (image_type_ == ImageType::RAW_FRAME) {
    strides_ = {width_ * pix_size};
    plane_num_ = 1;
    img_bytes_ = width_ * height_ * pix_size;

  } else if (image_format_ == ImageFormat::RGB_PLANAR ||
             image_format_ == ImageFormat::BGR_PLANAR) {
    uint32_t stride = width_ * pix_size;
    if (align_size_ > 1) {
      stride = ALIGN(stride, align_size_);
    }
    strides_ = {stride, stride, stride};
    plane_num_ = 3;
    img_bytes_ = height_ * stride * 3;

  } else if (image_format_ == ImageFormat::RGB_PACKED ||
             image_format_ == ImageFormat::BGR_PACKED) {
    uint32_t stride = width_ * pix_size * 3;
    if (align_size_ > 1) {
      stride = ALIGN(stride, align_size_);
    }
    strides_ = {stride};
    plane_num_ = 1;
    img_bytes_ = height_ * stride;

  } else if (image_format_ == ImageFormat::YUV420SP_UV ||
             image_format_ == ImageFormat::YUV420SP_VU) {
    uint32_t stride0 = width_ * pix_size;

    if (align_size_ > 1) {
      stride0 = ALIGN(stride0, align_size_);
    }
    strides_ = {stride0, stride0};
    plane_num_ = 2;
    img_bytes_ = height_ * stride0 * 2;
  } else if (image_format_ == ImageFormat::YUV420P_UV ||
             image_format_ == ImageFormat::YUV420P_VU) {
    uint32_t stride0 = width_ * pix_size;
    uint32_t stride12 = width_ * pix_size / 2;
    if (align_size_ > 1) {
      stride0 = ALIGN(stride0, align_size_);
      stride12 = ALIGN(stride12, align_size_);
    }
    strides_ = {stride0, stride12, stride12};
    plane_num_ = 3;
    img_bytes_ = height_ * stride0 + height_ * stride12 / 2;
  } else if (image_format_ == ImageFormat::YUV422SP_UV ||
             image_format_ == ImageFormat::YUV422SP_VU) {
    uint32_t stride0 = width_ * pix_size;
    if (align_size_ > 1) {
      stride0 = ALIGN(stride0, align_size_);
    }
    strides_ = {stride0, stride0};
    plane_num_ = 2;
    img_bytes_ = height_ * stride0 * 2;
  } else if (image_format_ == ImageFormat::YUV422P_UV ||
             image_format_ == ImageFormat::YUV422P_VU) {
    uint32_t stride0 = width_ * pix_size;
    if (align_size_ > 1) {
      stride0 = ALIGN(stride0, align_size_);
    }
    strides_ = {stride0, stride0, stride0};
    plane_num_ = 3;
    img_bytes_ = height_ * stride0 * 2;
  } else if (image_format_ == ImageFormat::GRAY) {
    uint32_t stride0 = width_ * pix_size;
    if (align_size_ > 1) {
      stride0 = ALIGN(stride0, align_size_);
    }
    strides_ = {stride0};
    plane_num_ = 1;
    img_bytes_ = height_ * stride0;
  } else {
    LOGE("Unsupported image format: %d", image_format_);
    return -1;
  }
  return 0;
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
  if (image_format_ == ImageFormat::RGB_PLANAR ||
      image_format_ == ImageFormat::BGR_PLANAR ||
      image_format_ == ImageFormat::GRAY) {
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
  memory_block_->own_memory = true;
  int32_t ret = setupMemoryBlock(memory_block_);
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

std::vector<uint64_t> BaseImage::getPhysicalAddress() const {
  if (image_type_ == ImageType::RAW_FRAME) {
    return {memory_block_->physicalAddress};
  }
  LOGE("get physical address failed");
  return {};
}

std::vector<uint8_t*> BaseImage::getVirtualAddress() const {
  if (image_type_ == ImageType::RAW_FRAME) {
    return {(uint8_t*)memory_block_->virtualAddress};
  }
  LOGE("get virtual address failed");
  return {};
}

int32_t BaseImage::setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                               uint32_t length) {
  LOGW("setup memory in BaseImage");
  return 0;
}

int32_t BaseImage::copyFromBuffer(const uint8_t* buffer, uint32_t size) {
  if (buffer == nullptr || size == 0) {
    LOGE("copy from buffer failed,buffer is nullptr or size is 0");
    return -1;
  }
  if (size != memory_block_->size) {
    LOGE(
        "copy from buffer failed,size is not "
        "equal,buffer_size:%d,memoryblock_size:%d",
        size, memory_block_->size);
    return -1;
  }
  memcpy(memory_block_->virtualAddress, buffer, size);
  int32_t ret = flushCache();
  if (ret != 0) {
    LOGE("flush cache failed");
    return ret;
  }
  return 0;
}