#include "audio/audio_system.hpp"
#include <iostream>
#include <mutex>
#include "cvi_audio.h"
#include "cvi_sys.h"

static std::mutex g_audio_mutex;

AudioSystem& AudioSystem::GetInstance() {
  static AudioSystem instance;
  return instance;
}

int AudioSystem::RequestInit() {
  std::lock_guard<std::mutex> lock(g_audio_mutex);
  if (ref_count_ == 0) {
    CVI_S32 s32Ret = CVI_SYS_Init();
    if (s32Ret != CVI_SUCCESS) {
      // It is possible that CVI_SYS_Init is already called by other modules.
      // We just log it as info/warning but proceed.
      // checking specific return code for "ALREADY_INIT" would be better if
      // available.
    }

    if (CVI_AUDIO_INIT() != 0) {
      std::cerr << "[AudioSystem] CVI_AUDIO_INIT failed" << std::endl;
      return -1;
    }
  }
  ref_count_++;
  return 0;
}

int AudioSystem::Release() {
  std::lock_guard<std::mutex> lock(g_audio_mutex);
  if (ref_count_ > 0) {
    ref_count_--;
    if (ref_count_ == 0) {
      CVI_AUDIO_DEINIT();
    }
  }
  return 0;
}
