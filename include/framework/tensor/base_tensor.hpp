#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H

#include <cassert>
#include <string>
#include <vector>

#include "common/common_types.hpp"
#include "image/base_image.hpp"
#include "memory/base_memory_pool.hpp"
class BaseTensor {
 public:
  // Constructors and Destructor
  BaseTensor(int element_bytes, std::shared_ptr<BaseMemoryPool> memory_pool);
  virtual ~BaseTensor();

  // Shape and Size Management
  void reshape(int n, int c, int h, int w, bool alloc_memory = true);

  virtual int32_t release();

  // Memory Management

  int32_t shareMemory(void* host_memory, uint64_t device_address,
                      const std::vector<int>& shape);

  // Shape and Size Query
  std::vector<int> getShape() const;
  int getNumElements() const;
  int getCapacity() const;
  int getElementSize() const;

  // Accessors for Dimensions
  uint32_t getWidth() const;
  uint32_t getHeight() const;
  int getChannels() const;
  int getBatchSize() const;

  MemoryBlock* getMemoryBlock();

  // Data Synchronization
  int32_t flushCache();
  int32_t invalidateCache();
  void setZero();

  int32_t constructImage(std::shared_ptr<BaseImage> image, int batch_idx = -1);
  int32_t copyFromImage(std::shared_ptr<BaseImage> image, int batch_idx = -1);

  template <typename T>
  T* getBatchPtr(int batch_idx) {
    if (sizeof(T) != element_bytes_) {
      printf(
          "element_bytes_ not equal to "
          "sizeof(T),element_bytes_:%d,sizeof(T):%ld",
          element_bytes_, sizeof(T));
      assert(0);
    }
    int num_elements = getNumElements();
    int batch_element_num = num_elements / shape_[0];
    return reinterpret_cast<T*>(memory_block_->virtualAddress) +
           batch_idx * batch_element_num;
  }

  // File I/O
  void dumpToFile(const std::string& file_path);
  void loadFromFile(const std::string& file_path);

  // Random Fill
  int32_t randomFill();

 protected:
  int element_bytes_;  // Size of each element in bytes
  bool owns_data_;

  // for soc mode, host_address_ is not nullptr
  std::unique_ptr<MemoryBlock> memory_block_;
  std::shared_ptr<BaseMemoryPool> memory_pool_;

  std::vector<int> shape_;

  // Deleted copy semantics
  BaseTensor(const BaseTensor&) = delete;
  BaseTensor& operator=(const BaseTensor&) = delete;
};

#endif  // BASE_TENSOR_H