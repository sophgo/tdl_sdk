#pragma once
#include "core/core.hpp"
#include "core/cviai_core.h"
#include "core/vpss_engine.hpp"
#include "deepsort/cvi_deepsort.hpp"
#include "ive/ive.h"
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
  std::vector<cviai_model_t> custom_cont;
  IVE_HANDLE ive_handle = NULL;
  std::vector<cviai::VpssEngine *> vec_vpss_engine;
  TamperDetectorMD *td_model = nullptr;
  Deepsort *ds_tracker = nullptr;
  bool use_gdc_wrap = false;
} cviai_context_t;

inline int __attribute__((always_inline)) GetModelName(cviai_model_t &model, char **filepath) {
  char *path = (char *)malloc(model.model_path.size());
  snprintf(path, model.model_path.size(), "%s", model.model_path.c_str());
  *filepath = path;
  return CVI_SUCCESS;
}

inline cviai::VpssEngine *__attribute__((always_inline))
CVI_AI_GetVpssEngine(cviai_handle_t handle, uint32_t index) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (index >= ctx->vec_vpss_engine.size()) {
    return nullptr;
  }
  return ctx->vec_vpss_engine[index];
}

inline int __attribute__((always_inline))
CVI_AI_AddVpssEngineThread(const uint32_t thread, const VPSS_GRP vpssGroupId, uint32_t *vpss_thread,
                           std::vector<cviai::VpssEngine *> *vec_engine) {
  *vpss_thread = thread;
  if (thread >= vec_engine->size()) {
    auto inst = new cviai::VpssEngine();
    if (inst->init(vpssGroupId) != CVI_SUCCESS) {
      LOGE("Vpss init failed\n");
      delete inst;
      return CVI_FAILURE;
    }

    vec_engine->push_back(inst);
    if (thread != vec_engine->size() - 1) {
      LOGW(
          "Thread %u is not in use, thus %u is changed to %u automatically. Used vpss group id is "
          "%u.\n",
          *vpss_thread, thread, *vpss_thread, inst->getGrpId());
      *vpss_thread = vec_engine->size() - 1;
    }
  } else {
    LOGW("Thread %u already exists, given group id %u will not be used.\n", thread, vpssGroupId);
  }
  return CVI_SUCCESS;
}

inline int __attribute__((always_inline))
setVPSSThread(cviai_model_t &model, std::vector<cviai::VpssEngine *> &v_engine,
              const uint32_t thread, const VPSS_GRP vpssGroupId) {
  uint32_t vpss_thread;
  if (int ret =
          CVI_AI_AddVpssEngineThread(thread, vpssGroupId, &vpss_thread, &v_engine) != CVI_SUCCESS) {
    return ret;
  }
  model.vpss_thread = vpss_thread;
  if (model.instance != nullptr) {
    model.instance->setVpssEngine(v_engine[model.vpss_thread]);
  }
  return CVI_SUCCESS;
}