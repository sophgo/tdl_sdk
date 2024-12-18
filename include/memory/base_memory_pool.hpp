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
#include "image/base_image.hpp"

class BaseMemoryPool {
 public:
  virtual ~BaseMemoryPool() = default;

  /**
   * 初始化内存池。
   * @param blockSize 单个内存块的大小（字节数）
   * @param initialBlockCount 初始化时的内存块数量
   */
  virtual bool initialize(uint32_t blockSize, uint32_t initialBlockCount) = 0;

  /**
   * 分配内存块。
   * @param size 请求的内存块大小
   * @return 成功分配时返回指向内存块的指针，否则返回空。
   */
  virtual std::unique_ptr<MemoryBlock> allocate(uint32_t size, uint32_t timeout_ms = 0) = 0;

  /**
   * 释放内存块。
   * @param block 需要回收的内存块
   */
  virtual bool recycle(std::unique_ptr<MemoryBlock> &block) = 0;

  virtual bool release(std::unique_ptr<MemoryBlock> &block) = 0;

  /**
   * 清空内存池，释放所有分配的资源。
   */
  virtual bool clear() = 0;

  /**
   * 获取内存池中总块数（包括空闲和已分配）。
   * @return 总块数。
   */
  virtual uint32_t totalBlocks() { return allocatedBlocks_.size(); };

 protected:
  std::unordered_map<uint32_t, BlockingQueue<std::unique_ptr<MemoryBlock>>>
      blockPools_;                              // 不同尺寸的内存块池
  std::vector<MemoryBlock *> allocatedBlocks_;  // 映射已分配的内存块地址到内存块信息
  int32_t device_id_ = 0;
};

#endif  // BASE_MEMORY_POOL_H