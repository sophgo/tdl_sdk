#include "core/cviai_custom.h"
#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"
#include "cviai_log.hpp"
#include "cviai_trace.hpp"

#include "custom/custom.hpp"
#include "cviai_experimental.h"

#include <string.h>

CVI_S32 CVI_AI_Custom_AddInference(cviai_handle_t handle, uint32_t *id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t model;
  model.instance = new cviai::Custom();
  ctx->custom_cont.push_back(model);
  *id = ctx->custom_cont.size() - 1;
  LOGI("Custom AI instance added.");
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_Custom_SetModelPath(cviai_handle_t handle, const uint32_t id, const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVIAI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init. Please call CVI_AI_Custom_CloseModel to reset.\n");
    return CVIAI_FAILURE;
  }
  mt.model_path = filepath;
  return CVIAI_SUCCESS;
}

const char *CVI_AI_Custom_GetModelPath(cviai_handle_t handle, const uint32_t id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return NULL;
  }
  return GetModelName(ctx->custom_cont[id]);
}

CVI_S32 CVI_AI_Custom_SetVpssThread(cviai_handle_t handle, const uint32_t id,
                                    const uint32_t thread) {
  return CVI_AI_Custom_SetVpssThread2(handle, id, thread, -1);
}

CVI_S32 CVI_AI_Custom_SetVpssThread2(cviai_handle_t handle, const uint32_t id,
                                     const uint32_t thread, const VPSS_GRP vpssGroupId) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVIAI_FAILURE;
  }
  return setVPSSThread(ctx->custom_cont[id], ctx->vec_vpss_engine, thread, vpssGroupId, 0);
}

CVI_S32 CVI_AI_Custom_GetVpssThread(cviai_handle_t handle, const uint32_t id, uint32_t *thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVIAI_FAILURE;
  }
  *thread = ctx->custom_cont[id].vpss_thread;
  return CVIAI_SUCCESS;
}

inline cviai::Custom *__attribute__((always_inline))
getCustomInstance(const uint32_t id, cviai_context_t *ctx) {
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return nullptr;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init. Please call CVI_AI_Custom_CloseModel to reset.\n");
    return nullptr;
  }
  return dynamic_cast<cviai::Custom *>(mt.instance);
}

CVI_S32 CVI_AI_Custom_SetVpssPreprocessParam(cviai_handle_t handle, const uint32_t id,
                                             const uint32_t inputIndex, const float *factor,
                                             const float *mean, const uint32_t length,
                                             const bool keepAspectRatio) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstance(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->setSQParam(inputIndex, factor, mean, length, true, keepAspectRatio);
}

CVI_S32 CVI_AI_Custom_SetVpssPreprocessParamRaw(cviai_handle_t handle, const uint32_t id,
                                                const uint32_t inputIndex, const float *qFactor,
                                                const float *qMean, const uint32_t length,
                                                const bool keepAspectRatio) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstance(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->setSQParam(inputIndex, qFactor, qMean, length, false, keepAspectRatio);
}

CVI_S32 CVI_AI_Custom_SetPreprocessFuncPtr(cviai_handle_t handle, const uint32_t id,
                                           preProcessFunc func, const bool use_tensor_input,
                                           const bool use_vpss_sq) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstance(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->setPreProcessFunc(func, use_tensor_input, use_vpss_sq);
}

CVI_S32 CVI_AI_Custom_CloseModel(cviai_handle_t handle, const uint32_t id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVIAI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  mt.instance->modelClose();
  return CVIAI_SUCCESS;
}

inline cviai::Custom *__attribute__((always_inline))
getCustomInstanceInit(const uint32_t id, cviai_context_t *ctx) {
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return nullptr;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized() == false) {
    if (mt.model_path.empty()) {
      LOGE("Model path for FaceAttribute is empty.\n");
      return nullptr;
    }
    if (mt.instance->modelOpen(mt.model_path.c_str()) != CVIAI_SUCCESS) {
      LOGE("Open model failed (%s).\n", mt.model_path.c_str());
      return nullptr;
    }
    mt.instance->setVpssEngine(ctx->vec_vpss_engine[mt.vpss_thread]);
  }
  return dynamic_cast<cviai::Custom *>(mt.instance);
}

CVI_S32 CVI_AI_Custom_GetInputTensorNCHW(cviai_handle_t handle, const uint32_t id,
                                         const char *tensorName, uint32_t *n, uint32_t *c,
                                         uint32_t *h, uint32_t *w) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstanceInit(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->getInputShape(tensorName, n, c, h, w);
}

CVI_S32 CVI_AI_Custom_RunInference(cviai_handle_t handle, const uint32_t id,
                                   VIDEO_FRAME_INFO_S *frame, uint32_t numOfFrames) {
  TRACE_EVENT("cviai_core", "CVI_AI_Custom_RetinaFace");
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstanceInit(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->inference(frame, numOfFrames);
}

CVI_S32 CVI_AI_Custom_GetOutputTensor(cviai_handle_t handle, const uint32_t id,
                                      const char *tensorName, int8_t **tensor,
                                      uint32_t *tensorCount, uint16_t *unitSize) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstanceInit(id, ctx);
  if (inst_ptr == nullptr) {
    return CVIAI_FAILURE;
  }
  return inst_ptr->getOutputTensor(tensorName, tensor, tensorCount, unitSize);
}
