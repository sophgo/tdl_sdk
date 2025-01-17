#include "common/device_info.hpp"
#include "bmruntime_interface.h"
std::vector<int> get_device_ids() {
  std::vector<int> devices;
  for (int i = 0; i < 36; i++) {
    bm_handle_t handle;
    bm_status_t st = bm_dev_request(&handle, i);

    if (st == BM_SUCCESS) {
      devices.push_back(i);
      bm_dev_free(handle);
    }
  }
  return devices;
}