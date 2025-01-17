#include <iostream>

#ifdef USE_TENSORRT
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "common/common.hpp"



BMContext::BMContext() {
}

BMContext::~BMContext() {
  for(auto kv:device_handles_){
    bm_dev_free(kv.second);
  }
  device_handles_.clear();  

}
BMContext& BMContext::Get() {
  static BMContext bm_ctx;
  return bm_ctx;
}

bm_handle_t BMContext::get_handle(int device_id){
  if(device_handles_.count(device_id)){
    return device_handles_[device_id];
  }
   
  if(device_id == -1){
    if(device_handles_.size()){
      for(auto kv:device_handles_){
        return kv.second;
      }
    }
    return nullptr;
  }
  pthread_mutex_lock(&lock_);
  bm_handle_t h;
  bm_dev_request(&h, device_id);
  device_handles_[device_id] = h;
  pthread_mutex_unlock(&lock_);
  return device_handles_[device_id];
}

void BMContext::set_device_id(int device_id){
  // BMContext& inst = Get();
  cnn_bm168x_handle(device_id);
}

