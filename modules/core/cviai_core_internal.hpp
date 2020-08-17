#pragma once
#include "core/cviai_core.h"

#include "core/utils/vpss_helper.h"
#include "vpss_engine.hpp"

#include "cviai_experimental.h"

#include "face_attribute/face_attribute.hpp"
#include "face_quality/face_quality.hpp"
#include "liveness/liveness.hpp"
#include "mask_classification/mask_classification.hpp"
#include "mask_face_recognition/mask_face_recognition.hpp"
#include "object_detection/mobiledetv2/mobiledetv2.hpp"
#include "object_detection/yolov3/yolov3.hpp"
#include "retina_face/retina_face.hpp"
#include "thermal_face_detection/thermal_face.hpp"

typedef struct {
  cviai::Core *instance = nullptr;
  std::string model_path = "";
  bool skip_vpss_preprocess = false;
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
  std::vector<cviai::VpssEngine *> vec_vpss_engine;
  bool use_gdc_wrap = false;
} cviai_context_t;

inline cviai::VpssEngine *CVI_AI_GetVpssEngine(cviai_handle_t *handle, uint32_t index) {
  cviai_context_t *ctx = new cviai_context_t;
  if (index >= ctx->vec_vpss_engine.size()) {
    return nullptr;
  }
  return ctx->vec_vpss_engine[index];
}