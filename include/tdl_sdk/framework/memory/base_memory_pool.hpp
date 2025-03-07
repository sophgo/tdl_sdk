#ifndef BASE_MEMORY_POOL_H
#define BASE_MEMORY_POOL_H

#include <cstddef>  // for size_t
#include <cstddef>
#include <cstdint>  // for uint64_t
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/blocking_queue.hpp"
#include "common/common_types.hpp"

enum class MemoryPoolType {
  CVI_SOC_DEVICE,
  BM_SOC_DEVICE,
};

class BaseMemoryPool {
 public:
  virtual ~BaseMemoryPool() = default;

  /**
   * 初始化内存池。
   * @param blockSize 单个内存块的大小（字节数）
   * @param initialBlockCount 初始化时的内存块数量
   */
  virtual int32_t initialize(uint32_t blockSize, uint32_t initialBlockCount);

  // 从缓存中获取内存块,假如缓存里没有,则新分配一块
  virtual std::unique_ptr<MemoryBlock> getBlock(uint32_t size,
                                                uint32_t timeout_ms = 0);

  // 回收内存块
  virtual int32_t recycle(std::unique_ptr<MemoryBlock> &block);

  // 分配内存块
  virtual std::unique_ptr<MemoryBlock> allocate(uint32_t size,
                                                uint32_t timeout_ms = 0) = 0;
  // 释放内存块,不放回换成池
  virtual int32_t release(std::unique_ptr<MemoryBlock> &block) = 0;

  virtual int32_t invalidateCache(std::unique_ptr<MemoryBlock> &block) = 0;
  virtual int32_t flushCache(std::unique_ptr<MemoryBlock> &block) = 0;
  virtual uint32_t totalBlocks() { return allocatedBlocks_.size(); };
  virtual int32_t clear();

 protected:
  std::unordered_map<uint32_t, BlockingQueue<std::unique_ptr<MemoryBlock>>>
      blockPools_;  // 不同尺寸的内存块池
  std::vector<MemoryBlock *>
      allocatedBlocks_;  // 映射已分配的内存块地址到内存块信息
  int32_t device_id_ = 0;
};

class BaseMemoryPoolFactory {
 public:
  static std::shared_ptr<BaseMemoryPool> createMemoryPool(
      MemoryPoolType memory_pool_type);
};

#endif  // BASE_MEMORY_POOL_H