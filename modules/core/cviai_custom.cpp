#include "core/cviai_custom.h"
#include "core/cviai_core.h"
#include "cviai_core_internal.hpp"
#include "cviai_log.hpp"

#include <string.h>
#include "custom/custom.hpp"
#include "cviai_experimental.h"
#include "cviai_perfetto.h"
#include "cviai_trace.hpp"

int CVI_AI_Custom_AddInference(cviai_handle_t handle, uint32_t *id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t model;
  model.instance = new cviai::Custom();
  ctx->custom_cont.push_back(model);
  *id = ctx->custom_cont.size() - 1;
  return CVI_SUCCESS;
}

int CVI_AI_Custom_SetModelPath(cviai_handle_t handle, const uint32_t id, const char *filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init.\n");
    return CVI_FAILURE;
  }
  mt.model_path = filepath;
  return CVI_SUCCESS;
}

int CVI_AI_Custom_GetModelPath(cviai_handle_t handle, const uint32_t id, char **filepath) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  char *path = (char *)malloc(mt.model_path.size());
  snprintf(path, strlen(path), "%s", mt.model_path.c_str());
  *filepath = path;
  return CVI_SUCCESS;
}

int CVI_AI_Custom_SetVpssThread(cviai_handle_t handle, const uint32_t id, const uint32_t thread) {
  return CVI_AI_Custom_SetVpssThread2(handle, id, thread, -1);
}

int CVI_AI_Custom_SetVpssThread2(cviai_handle_t handle, const uint32_t id, const uint32_t thread,
                                 const VPSS_GRP vpssGroupId) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  uint32_t vpss_thread;
  if (int ret = CVI_AI_AddVpssEngineThread(thread, vpssGroupId, &vpss_thread,
                                           &ctx->vec_vpss_engine) != CVI_SUCCESS) {
    return ret;
  }
  auto &m_t = ctx->custom_cont[id];
  m_t.vpss_thread = vpss_thread;
  if (m_t.instance != nullptr) {
    m_t.instance->setVpssEngine(ctx->vec_vpss_engine[m_t.vpss_thread]);
  }
  return CVI_SUCCESS;
}

int CVI_AI_Custom_GetVpssThread(cviai_handle_t handle, const uint32_t id, uint32_t *thread) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  *thread = ctx->custom_cont[id].vpss_thread;
  return CVI_SUCCESS;
}

int CVI_AI_Custom_SetVpssPreprocessParam(cviai_handle_t handle, const uint32_t id,
                                         const float *factor, const float *mean,
                                         const uint32_t length, const float quantizationThreshold,
                                         const bool keepAspectRatio) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init.\n");
    return CVI_FAILURE;
  }
  auto *inst_ptr = dynamic_cast<cviai::Custom *>(mt.instance);
  return inst_ptr->setSQParam(factor, mean, length, quantizationThreshold, keepAspectRatio);
}

int CVI_AI_Custom_SetVpssPreprocessParamRaw(cviai_handle_t handle, const uint32_t id,
                                            const float *qFactor, const float *qMean,
                                            const uint32_t length, const bool keepAspectRatio) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init.\n");
    return CVI_FAILURE;
  }
  auto *inst_ptr = dynamic_cast<cviai::Custom *>(mt.instance);
  return inst_ptr->setSQParamRaw(qFactor, qMean, length, keepAspectRatio);
}

int CVI_AI_Custom_SetPreprocessFuncPtr(cviai_handle_t handle, const uint32_t id,
                                       preProcessFunc func, const bool use_tensor_input,
                                       const bool use_vpss_sq) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init.\n");
    return CVI_FAILURE;
  }
  auto *inst_ptr = dynamic_cast<cviai::Custom *>(mt.instance);
  return inst_ptr->setPreProcessFunc(func, use_tensor_input, use_vpss_sq);
}

int CVI_AI_Custom_SetSkipPostProcess(cviai_handle_t handle, const uint32_t id, const bool skip) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized()) {
    LOGE("Inference already init.\n");
    return CVI_FAILURE;
  }
  auto *inst_ptr = dynamic_cast<cviai::Custom *>(mt.instance);
  return inst_ptr->setSkipPostProcess(skip);
}

inline cviai::Custom *__attribute__((always_inline))
getCustomInstance(const uint32_t id, cviai_context_t *ctx) {
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
    if (mt.instance->modelOpen(mt.model_path.c_str()) != CVI_SUCCESS) {
      LOGE("Open model failed (%s).\n", mt.model_path.c_str());
      return nullptr;
    }
    mt.instance->setVpssEngine(ctx->vec_vpss_engine[mt.vpss_thread]);
  }
  return dynamic_cast<cviai::Custom *>(mt.instance);
}

int CVI_AI_Custom_GetInputTensorNCHW(cviai_handle_t handle, const uint32_t id,
                                     const char *tensorName, uint32_t *n, uint32_t *c, uint32_t *h,
                                     uint32_t *w) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstance(id, ctx);
  if (inst_ptr == nullptr) {
    return CVI_FAILURE;
  }
  return inst_ptr->getNCHW(tensorName, n, c, h, w);
}

int CVI_AI_Custom_RunInference(cviai_handle_t handle, const uint32_t id,
                               VIDEO_FRAME_INFO_S *frame) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  auto *inst_ptr = getCustomInstance(id, ctx);
  if (inst_ptr == nullptr) {
    return CVI_FAILURE;
  }
  return inst_ptr->inference(frame);
}

int CVI_AI_Custom_GetOutputTensor(cviai_handle_t handle, const uint32_t id, const char *tensorName,
                                  int8_t **tensor, uint32_t *tensorCount, uint16_t *unitSize) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  if (id >= (uint32_t)ctx->custom_cont.size()) {
    LOGE("Exceed id, given %d, total %zu.\n", id, ctx->custom_cont.size());
    return CVI_FAILURE;
  }
  cviai_model_t &mt = ctx->custom_cont[id];
  if (mt.instance->isInitialized() == false) {
    LOGE("Inference not init.\n");
    return CVI_FAILURE;
  }
  auto *inst_ptr = dynamic_cast<cviai::Custom *>(mt.instance);
  return inst_ptr->getOutputTensor(tensorName, tensor, tensorCount, unitSize);
}

int CVI_AI_Custom_CloseModel(cviai_handle_t handle, const uint32_t id) {
  cviai_context_t *ctx = static_cast<cviai_context_t *>(handle);
  cviai_model_t &m_t = ctx->custom_cont[id];
  if (m_t.instance == nullptr) {
    return CVI_FAILURE;
  }
  m_t.instance->modelClose();
  return CVI_SUCCESS;
}
