#pragma once
#include "core/cviai_core.h"

#include "core/core.hpp"
#include "ive/ive.h"
#include "vpss_engine.hpp"

#include "tamper_detection/tamper_detection.hpp"

#include <unordered_map>

typedef struct {
  cviai::Core *instance = nullptr;
  std::string model_path = "";
  bool skip_vpss_preprocess = false;
  float model_threshold = -1;
  uint32_t vpss_thread = 0;
} cviai_model_t;

// specialize std::hash for enum CVI_AI_SUPPORTED_MODEL_E
namespace std {
template <>
struct hash<CVI_AI_SUPPORTED_MODEL_E> {
  size_t operator()(CVI_AI_SUPPORTED_MODEL_E value) const { return static_cast<size_t>(value); }
};
}  // namespace std

typedef struct {
  std::unordered_map<CVI_AI_SUPPORTED_MODEL_E, cviai_model_t> model_cont;
  IVE_HANDLE ive_handle = NULL;
  std::vector<cviai::VpssEngine *> vec_vpss_engine;
  TamperDetectorMD *td_model = nullptr;
  bool use_gdc_wrap = false;
} cviai_context_t;

inline cviai::VpssEngine *CVI_AI_GetVpssEngine(cviai_handle_t handle, uint32_t index) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (index >= ctx->vec_vpss_engine.size()) {
    return nullptr;
  }
  return ctx->vec_vpss_engine[index];
}