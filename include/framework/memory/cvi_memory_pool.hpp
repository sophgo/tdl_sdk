#ifndef CVI_MEMORY_POOL_H
#define CVI_MEMORY_POOL_H

#include "memory/base_memory_pool.hpp"

class CviMemoryPool : public BaseMemoryPool {
 public:
  CviMemoryPool();
  ~CviMemoryPool();

  std::unique_ptr<MemoryBlock> allocate(uint32_t size,
                                        uint32_t timeout_ms = 10) override;

  int32_t release(std::unique_ptr<MemoryBlock> &block) override;

  std::unique_ptr<MemoryBlock> CreateExVb(uint32_t blk_size,
                                          uint32_t blk_cnt,
                                          uint32_t weight,
                                          uint32_t height);
  int32_t DestroyExVb(std::unique_ptr<MemoryBlock> &block);

  virtual int32_t flushCache(std::unique_ptr<MemoryBlock> &block) override;
  virtual int32_t invalidateCache(std::unique_ptr<MemoryBlock> &block) override;
  //   bool allocateImage(std::shared_ptr<BaseImage> &image) override;

 private:
  int32_t num_allocated_ = 0;
  std::string str_mem_pool_name_ = "cvi_mem_pool";
};

#endif  // CVI_MEMORY_POOL_H