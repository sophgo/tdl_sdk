#include "memory/cpu_memory_pool.hpp"

#include "utils/tdl_log.hpp"
CpuMemoryPool::CpuMemoryPool() { LOGI("CpuMemoryPool constructor"); }

CpuMemoryPool::~CpuMemoryPool() { LOGI("CpuMemoryPool destructor"); }

std::unique_ptr<MemoryBlock> CpuMemoryPool::allocate(uint32_t size,
                                                     uint32_t timeout_ms) {
  std::unique_ptr<MemoryBlock> block = std::make_unique<MemoryBlock>();
  block->size = size;
  block->virtualAddress = new uint8_t[size];
  block->physicalAddress = 0;
  block->own_memory = true;
  return block;
}

int32_t CpuMemoryPool::release(std::unique_ptr<MemoryBlock> &block) {
  if (block->own_memory) {
    delete[] (uint8_t *)block->virtualAddress;
  }
  block->virtualAddress = nullptr;
  block->size = 0;
  block->physicalAddress = 0;
  block->own_memory = false;
  return 0;
}

int32_t CpuMemoryPool::flushCache(std::unique_ptr<MemoryBlock> &block) {
  return 0;
}

int32_t CpuMemoryPool::invalidateCache(std::unique_ptr<MemoryBlock> &block) {
  return 0;
}
