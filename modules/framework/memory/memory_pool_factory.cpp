#include "memory/base_memory_pool.hpp"
#ifdef __SOPHON__
#include "memory/bm_memory_pool.hpp"
#else
#include "memory/cvi_memory_pool.hpp"
#endif

std::shared_ptr<BaseMemoryPool> BaseMemoryPoolFactory::createMemoryPool(
    MemoryPoolType memory_pool_type) {
  switch (memory_pool_type) {
    case MemoryPoolType::CVI_SOC_DEVICE:
#ifndef __SOPHON__
      return std::make_shared<CviMemoryPool>();
#else
      return nullptr;
#endif
    case MemoryPoolType::BM_SOC_DEVICE:
#ifdef __SOPHON__
      return std::make_shared<BmMemoryPool>(nullptr);
#else
      return nullptr;
#endif
    default:
      return nullptr;
  }
}
