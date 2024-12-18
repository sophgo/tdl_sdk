#ifndef CVI_MEMORY_POOL_H
#define CVI_MEMORY_POOL_H

#include "memory/base_memory_pool.hpp"

class CviMemoryPool : public BaseMemoryPool {
 public:
  CviMemoryPool();
  ~CviMemoryPool();

  bool initialize(uint32_t blockSize, uint32_t initialBlockCount) override;
  std::unique_ptr<MemoryBlock> allocate(uint32_t size, uint32_t timeout_ms = 10) override;
  std::unique_ptr<MemoryBlock> allocate_impl(uint32_t size);
  bool recycle(std::unique_ptr<MemoryBlock> &block) override;
  bool release(std::unique_ptr<MemoryBlock> &block) override;
  bool clear() override;
  static std::unique_ptr<MemoryBlock> create(uint32_t size);
  static std::unique_ptr<MemoryBlock> create_vb(uint32_t size);
  //   bool allocateImage(std::shared_ptr<BaseImage> &image) override;

 private:
  int32_t num_allocated_ = 0;
  std::string str_mem_pool_name_ = "cvi_mem_pool";
};

#endif  // CVI_MEMORY_POOL_H