#include "qwen.hpp"
#include <dlfcn.h>
#include "utils/tdl_log.hpp"

Qwen::Qwen() : decrypt_handle_(nullptr), decrypt_func_(nullptr) {}

Qwen::~Qwen() { deinit_decrypt(); }

void Qwen::init_decrypt() {
  if (lib_path_.empty()) {
    return;
  }

  decrypt_handle_ = dlopen(lib_path_.c_str(), RTLD_LAZY);
  if (!decrypt_handle_) {
    LOGE("Decrypt lib [%s] load failed.", lib_path_.c_str());
    return;
  }

  decrypt_func_ = (decltype(decrypt_func_))dlsym(decrypt_handle_, "decrypt");
  auto error = dlerror();
  if (error) {
    dlclose(decrypt_handle_);
    LOGE("Decrypt lib [%s] symbol find failed.", lib_path_.c_str());
    return;
  }
}

void Qwen::deinit_decrypt() {
  if (lib_path_.empty()) {
    return;
  }

  if (decrypt_handle_) {
    dlclose(decrypt_handle_);
    decrypt_handle_ = nullptr;
  }
  decrypt_func_ = nullptr;
}

int32_t Qwen::onModelOpened() {
  init_decrypt();
  return 0;
}

int32_t Qwen::onModelClosed() {
  deinit_decrypt();
  return 0;
}