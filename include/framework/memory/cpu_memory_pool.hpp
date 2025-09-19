#ifndef CPU_MEMORY_POOL_H
#define CPU_MEMORY_POOL_H

#include "memory/base_memory_pool.hpp"

class CpuMemoryPool : public BaseMemoryPool {
 public:
  CpuMemoryPool();
  ~CpuMemoryPool();

  std::unique_ptr<MemoryBlock> allocate(uint32_t size,
                                        uint32_t timeout_ms = 10) override;

  int32_t release(std::unique_ptr<MemoryBlock> &block) override;

  static std::unique_ptr<MemoryBlock> create_vb(uint32_t size);
  virtual int32_t flushCache(std::unique_ptr<MemoryBlock> &block) override;
  virtual int32_t invalidateCache(std::unique_ptr<MemoryBlock> &block) override;
  //   bool allocateImage(std::shared_ptr<BaseImage> &image) override;

 private:
  int32_t num_allocated_ = 0;
  std::string str_mem_pool_name_ = "cpu_mem_pool";
};

#endif  // CPU_MEMORY_POOL_H