#include "memory/base_memory_pool.hpp"
#include "memory/bm_memory_pool.hpp"
#include "memory/cvi_memory_pool.hpp"
std::shared_ptr<BaseMemoryPool> BaseMemoryPoolFactory::createMemoryPool(
    MemoryPoolType memory_pool_type) {
  switch (memory_pool_type) {
    case MemoryPoolType::CVI_SOC_DEVICE:
      return std::make_shared<CviMemoryPool>();
    case MemoryPoolType::BM_SOC_DEVICE:
      return std::make_shared<BmMemoryPool>(nullptr);
    default:
      return nullptr;
  }
}
