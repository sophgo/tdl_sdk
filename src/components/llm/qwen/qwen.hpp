#ifndef TDL_SDK_COMPONENTS_LLM_QWEN_QWEN_HPP
#define TDL_SDK_COMPONENTS_LLM_QWEN_QWEN_HPP

#include "model/llm_model.hpp"

class Qwen : public LLMModel {
 public:
  Qwen();
  virtual ~Qwen();

  // 重写基类的回调方法
  virtual int32_t onModelOpened() override;
  virtual int32_t onModelClosed() override;

 private:
  void init_decrypt();
  void deinit_decrypt();

 private:
  std::string lib_path_;
  void* decrypt_handle_;  // handle of decrypt lib
  uint8_t* (*decrypt_func_)(const uint8_t*, uint64_t,
                            uint64_t*);  // decrypt func from lib
};

#endif  // QWEN_HPP