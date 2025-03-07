#ifndef BM_MEMORY_POOL_H
#define BM_MEMORY_POOL_H

#include <bmruntime_interface.h>

#include <map>

#include "memory/base_memory_pool.hpp"

class BMContext {
 public:
  BMContext();
  ~BMContext();
  static BMContext &Get();

  bm_handle_t get_handle(int device_id);
  inline static bm_handle_t cnn_bm168x_handle(int device_id) {
    return Get().get_handle(device_id);
  }

  static void set_device_id(int device_id);

  std::map<int, bm_handle_t> device_handles_;

  pthread_mutex_t lock_ = PTHREAD_MUTEX_INITIALIZER;

 private:
  BMContext(const BMContext &) = delete;
  BMContext &operator=(const BMContext &) = delete;
};

class BmMemoryPool : public BaseMemoryPool {
 public:
  BmMemoryPool(void *bm_handle);
  virtual ~BmMemoryPool();

  std::unique_ptr<MemoryBlock> allocate(uint32_t size,
                                        uint32_t timeout_ms = 10) override;
  int32_t release(std::unique_ptr<MemoryBlock> &block) override;

  virtual int32_t flushCache(std::unique_ptr<MemoryBlock> &block) override;
  virtual int32_t invalidateCache(std::unique_ptr<MemoryBlock> &block) override;

 private:
  void *bm_handle_;
};

#endif  // BM_MEMORY_POOL_H
