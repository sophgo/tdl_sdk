#include "service/cviai_objservice.h"

#include "area_detect/area_detect.hpp"
#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"
#include "feature_matching/feature_matching.hpp"

#include <cvimath/cvimath.h>

typedef struct {
  cvai_service_feature_array_ext_t feature_array_ext;
  cviai_handle_t ai_handle = NULL;
  cviai::service::DigitalTracking *m_dt = nullptr;
  cviai::service::AreaDetect *m_ad = nullptr;
} cviai_objservice_context_t;

CVI_S32 CVI_AI_OBJService_CreateHandle(cviai_objservice_handle_t *handle,
                                       cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    LOGE("ai_handle is empty.");
    return CVI_FAILURE;
  }
  cviai_objservice_context_t *ctx = new cviai_objservice_context_t;
  ctx->ai_handle = ai_handle;
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_OBJService_DestroyHandle(cviai_objservice_handle_t handle) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  delete ctx->m_dt;
  delete ctx->m_ad;
  delete ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_OBJService_RegisterFeatureArray(cviai_objservice_handle_t handle,
                                               const cvai_service_feature_array_t featureArray) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  return RegisterFeatureArray(featureArray, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_OBJService_ObjectInfoMatching(cviai_objservice_handle_t handle,
                                             const cvai_object_info_t *object_info,
                                             const uint32_t k, uint32_t **index) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  return FeatureMatchingRaw((uint8_t *)object_info->feature.ptr, object_info->feature.type, k,
                            index, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_OBJService_RawMatching(cviai_objservice_handle_t handle, const uint8_t *feature,
                                      const feature_type_e type, const uint32_t k,
                                      uint32_t **index) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  return FeatureMatchingRaw(feature, type, k, index, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_OBJService_DigitalZoom(cviai_objservice_handle_t handle,
                                      const VIDEO_FRAME_INFO_S *inFrame, const cvai_object_t *meta,
                                      const float obj_skip_ratio, const float trans_ratio,
                                      VIDEO_FRAME_INFO_S *outFrame) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, obj_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_OBJService_DrawRect(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame) {
  return cviai::service::DrawMeta(meta, frame);
}

CVI_S32 CVI_AI_OBJService_SetIntersect(cviai_objservice_handle_t handle,
                                       const VIDEO_FRAME_INFO_S *frame, const cvai_pts_t *pts) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->m_ad == nullptr) {
    ctx->m_ad = new cviai::service::AreaDetect();
  }
  return ctx->m_ad->setArea(frame, *pts);
}

CVI_S32 CVI_AI_OBJService_DetectIntersect(cviai_objservice_handle_t handle,
                                          const VIDEO_FRAME_INFO_S *frame,
                                          const area_detect_pts_t *input,
                                          const uint32_t input_length,
                                          cvai_area_detect_e **status) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->m_ad == nullptr) {
    return CVI_FAILURE;
  }
  int ret = CVI_SUCCESS;
  std::vector<cvai_area_detect_e> stat;
  if ((ret = ctx->m_ad->run(frame, input, input_length, &stat)) == CVI_SUCCESS) {
    *status = (cvai_area_detect_e *)malloc(input_length * sizeof(cvai_area_detect_e));
    memcpy(*status, stat.data(), input_length * sizeof(cvai_area_detect_e));
  }
  return ret;
}