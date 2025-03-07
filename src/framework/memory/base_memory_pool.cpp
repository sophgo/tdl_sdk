#include "memory/base_memory_pool.hpp"

int32_t BaseMemoryPool::initialize(uint32_t blockSize,
                                   uint32_t initialBlockCount) {
  for (uint32_t i = 0; i < initialBlockCount; i++) {
    auto block = allocate(blockSize);
    if (block == nullptr) {
      return -1;
    }
    blockPools_[blockSize].push(std::move(block));
  }
  return 0;
}

std::unique_ptr<MemoryBlock> BaseMemoryPool::getBlock(uint32_t size,
                                                      uint32_t timeout_ms) {
  if (blockPools_.find(size) == blockPools_.end()) {
    return allocate(size, timeout_ms);
  }
  return blockPools_[size].pop(timeout_ms);
}

int32_t BaseMemoryPool::recycle(std::unique_ptr<MemoryBlock> &block) {
  if (block == nullptr) {
    return -1;
  }
  blockPools_[block->size].push(std::move(block));
  return 0;
}

int32_t BaseMemoryPool::clear() {
  for (auto &pool : blockPools_) {
    while (pool.second.size() > 0) {
      auto block = pool.second.pop();
      release(block);
    }
  }
  blockPools_.clear();
  return 0;
}
