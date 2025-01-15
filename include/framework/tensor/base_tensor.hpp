#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H

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
  void reshape(int n, int c, int h, int w);

  virtual int32_t release();

  // Memory Management

  void shareMemory(void* host_memory, uint64_t device_address, int element_size,
                   const std::vector<int>& shape);

  // Shape and Size Query
  std::vector<int> getShape() const;
  int getNumElements() const;
  int getCapacity() const;
  int getElementSize() const;

  // Accessors for Dimensions
  int getWidth() const;
  int getHeight() const;
  int getChannels() const;
  int getBatchSize() const;

  MemoryBlock* getMemoryBlock();

  // Data Synchronization
  int32_t flushCache();
  int32_t invalidateCache();
  void setZero();
  void syncWith(const BaseTensor& other);

  int32_t constructImage(std::shared_ptr<BaseImage> image, int batch_idx = -1);
  int32_t copyFromImage(std::shared_ptr<BaseImage> image, int batch_idx = -1);

  // File I/O
  void saveToFile(const std::string& file_path) const;
  void loadFromFile(const std::string& file_path);

 protected:
  // for soc mode, host_address_ is not nullptr
  std::unique_ptr<MemoryBlock> memory_block_;
  std::shared_ptr<BaseMemoryPool> memory_pool_;

  std::vector<int> shape_;
  int num_elements_;
  int element_bytes_;  // Size of each element in bytes
  bool owns_data_;

  // Deleted copy semantics
  BaseTensor(const BaseTensor&) = delete;
  BaseTensor& operator=(const BaseTensor&) = delete;
};

#endif  // BASE_TENSOR_H