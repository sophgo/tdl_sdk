#include "memory/base_memory_pool.hpp"
#ifdef __BM168X__
#include "memory/bm_memory_pool.hpp"
#elif defined(__CMODEL_CV181X__) || defined(__CMODEL_CV184X__)
#include "memory/cpu_memory_pool.hpp"
#else
#include "memory/cvi_memory_pool.hpp"
#endif

std::shared_ptr<BaseMemoryPool> BaseMemoryPoolFactory::createMemoryPool() {
#if defined(__BM168X__)
  return std::make_shared<BmMemoryPool>(nullptr);

#elif defined(__CV180X__) || defined(__CV181X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  return std::make_shared<CviMemoryPool>();

#elif defined(__CMODEL_CV181X__) || defined(__CMODEL_CV184X__)
  return std::make_shared<CpuMemoryPool>();
#else
  LOGE("Unsupported platform");
  return nullptr;
#endif
}
