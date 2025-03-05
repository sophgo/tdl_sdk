#include "tensor/base_tensor.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

#include "utils/tdl_log.hpp"

BaseTensor::BaseTensor(int element_bytes,
                       std::shared_ptr<BaseMemoryPool> memory_pool)
    : element_bytes_(element_bytes), num_elements_(0), owns_data_(false) {
  shape_.resize(4, 0);
  memory_block_ = nullptr;
  memory_pool_ = memory_pool;
}

BaseTensor::~BaseTensor() { release(); }

void BaseTensor::reshape(int n, int c, int h, int w) {
  shape_ = {n, c, h, w};
  int capacity = n * c * h * w * element_bytes_;
  if (memory_block_ == nullptr) {
    memory_block_ = memory_pool_->allocate(capacity);
  } else if (memory_block_->size < capacity) {
    memory_pool_->release(memory_block_);
    memory_block_ = memory_pool_->allocate(capacity);
  } else {
    // do nothing
  }
}

void BaseTensor::shareMemory(void* host_memory, uint64_t device_address,
                             int element_size, const std::vector<int>& shape) {
  if (owns_data_) {
    LOGE("host_memory is not nullptr, cannot share memory\n");
    return;
  }
  if (shape.size() != 4) {
    LOGE("shape size must be 4\n");
    return;
  }
  memory_block_ = std::make_unique<MemoryBlock>();
  memset(memory_block_.get(), 0, sizeof(MemoryBlock));
  memory_block_->virtualAddress = host_memory;
  memory_block_->physicalAddress = device_address;
  memory_block_->size =
      shape[0] * shape[1] * shape[2] * shape[3] * element_bytes_;
  shape_ = shape;
  num_elements_ =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  memory_block_->own_memory = false;
}

std::vector<int> BaseTensor::getShape() const { return shape_; }

int BaseTensor::getNumElements() const {
  return shape_[0] * shape_[1] * shape_[2] * shape_[3];
}

int BaseTensor::getCapacity() const {
  return shape_[0] * shape_[1] * shape_[2] * shape_[3] * element_bytes_;
}

int BaseTensor::getElementSize() const { return element_bytes_; }

int BaseTensor::getWidth() const { return shape_[3]; }

int BaseTensor::getHeight() const { return shape_[2]; }

int BaseTensor::getChannels() const { return shape_[1]; }

int BaseTensor::getBatchSize() const { return shape_[0]; }

void BaseTensor::dumpToFile(const std::string& file_path) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path
              << " for writing.\n";
    return;
  }

  if (memory_block_ == nullptr) {
    std::cerr << "Error: Host memory is not allocated.\n";
    return;
  }
  int32_t ret = invalidateCache();
  if (ret != 0) {
    LOGE("invalidateCache failed, ret: %d\n", ret);
    return;
  }
  int capacity = getCapacity();
  file.write(reinterpret_cast<const char*>(memory_block_->virtualAddress),
             capacity);
  file.close();
}

void BaseTensor::loadFromFile(const std::string& file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << file_path
              << " for reading.\n";
    return;
  }
  if (memory_block_ == nullptr) {
    std::cerr << "Error: Host memory is not allocated.\n";
    return;
  }
  int capacity = getCapacity();
  file.read(reinterpret_cast<char*>(memory_block_->virtualAddress), capacity);
  file.close();
}

int32_t BaseTensor::constructImage(std::shared_ptr<BaseImage> image,
                                   int batch_idx) {
  if (!image->isPlanar()) {
    LOGE("image is not planar\n");
    return -1;
  }
  if (!image->isAligned()) {
    LOGE("image is not aligned\n");
    return -1;
  }
  std::vector<uint32_t> strides = image->getStrides();

  uint32_t image_size = image->getImageByteSize();
  uint32_t min_tensor_size = (batch_idx + 1) * image_size;
  if (getCapacity() < min_tensor_size) {
    LOGE("tensor capacity(%d) < image size(%d)\n", getCapacity(),
         min_tensor_size);
    return -1;
  }
  uint64_t img_addr = memory_block_->physicalAddress + batch_idx * image_size;
  uint8_t* img_ptr = static_cast<uint8_t*>(memory_block_->virtualAddress) +
                     batch_idx * image_size;
  std::unique_ptr<MemoryBlock> memory_block = std::make_unique<MemoryBlock>();
  memory_block->physicalAddress = img_addr;
  memory_block->virtualAddress = img_ptr;
  memory_block->size = image_size;
  memory_block->own_memory = false;

  LOGI("constructImage, img_addr:%llu, img_ptr:%p, size:%d", img_addr, img_ptr,
       image_size);
  int32_t ret = image->setupMemoryBlock(memory_block);
  if (ret != 0) {
    LOGE("image setAddrInfo failed, ret: %d\n", ret);
    return -1;
  }
  return 0;
}

int32_t BaseTensor::copyFromImage(std::shared_ptr<BaseImage> image,
                                  int batch_idx) {
  if (image->getWidth() != getWidth() || image->getHeight() != getHeight()) {
    LOGE(
        "image width(%d) != tensor width(%d) or height(%d) != tensor "
        "height(%d)\n",
        image->getWidth(), getWidth(), image->getHeight(), getHeight());
    return -1;
  }
  if (batch_idx >= shape_[0]) {
    LOGE("batch_idx(%d) >= batch_size(%d)\n", batch_idx, shape_[0]);
    return -1;
  }
  uint32_t batch_bytes = shape_[1] * shape_[2] * shape_[3] * element_bytes_;

  uint8_t* dst_tensor_ptr =
      static_cast<uint8_t*>(memory_block_->virtualAddress) +
      batch_idx * batch_bytes;
  std::vector<uint8_t*> src_ptrs = image->getVirtualAddress();
  uint32_t plane_size = getWidth() * getHeight() * element_bytes_;
  uint32_t w = getWidth();
  uint32_t h = getHeight();

  LOGI(
      "copyFromImage, batch_idx:%d,img_stride:[%d,%d,%d], "
      "batch_bytes:%d,plane_size:%d,w:%d,h:%d,plane_num:%d,src_ptrs:%p,dst_ptr:"
      "%p",
      batch_idx, image->getStrides()[0], image->getStrides()[1],
      image->getStrides()[2], batch_bytes, plane_size, w, h,
      image->getPlaneNum(), src_ptrs[0], dst_tensor_ptr);
  LOGI(
      "tensor "
      "bytes:%d,memory_block_bytes:%d,memory_start:%p,memory_end:%p,element_"
      "bytes:%d",
      getCapacity(), memory_block_->size, memory_block_->virtualAddress,
      (uint8_t*)(memory_block_->virtualAddress) + memory_block_->size,
      element_bytes_);
  for (int i = 0; i < image->getPlaneNum(); i++) {
    uint8_t* src_ptr = src_ptrs[i];
    uint8_t* dst_ptr = dst_tensor_ptr + i * plane_size;
    uint32_t img_stride_i = image->getStrides()[i];
    if (img_stride_i == w * element_bytes_) {
      memcpy(dst_ptr, src_ptr, plane_size);
    } else {
      LOGI("plane:%d,src_ptr:%p,dst_ptr:%p,img_stride_i:%d", i, src_ptr,
           dst_ptr, img_stride_i);
      for (int j = 0; j < h; j++) {
        uint8_t* src_row_ptr = src_ptr + j * img_stride_i;
        uint8_t* dst_row_ptr = dst_ptr + j * w * element_bytes_;

        memcpy(dst_row_ptr, src_row_ptr, w * element_bytes_);
      }
    }
  }
  return 0;
}

int32_t BaseTensor::release() {
  if (memory_block_ != nullptr) {
    memory_pool_->release(memory_block_);
    memory_block_ = nullptr;
  }
  num_elements_ = 0;
  shape_.clear();
  owns_data_ = false;
  return 0;
}

int32_t BaseTensor::invalidateCache() {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr\n");
    return -1;
  }
  int32_t ret = memory_pool_->invalidateCache(memory_block_);
  return ret;
}

int32_t BaseTensor::flushCache() {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr\n");
    return -1;
  }
  int32_t ret = memory_pool_->flushCache(memory_block_);
  return ret;
}

MemoryBlock* BaseTensor::getMemoryBlock() {
  if (memory_block_ == nullptr) {
    return nullptr;
  }
  return memory_block_.get();
}
