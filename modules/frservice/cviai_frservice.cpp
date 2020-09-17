#include "service/cviai_frservice.h"

#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"
#include "feature_matching/feature_matching.hpp"

#include <string.h>
#include <syslog.h>

typedef struct {
  cvai_service_feature_array_ext_t feature_array_ext;
  cviai_handle_t ai_handle = NULL;
  cviai::service::DigitalTracking *m_dt = nullptr;
} cviai_frservice_context_t;

CVI_S32 CVI_AI_FRService_CreateHandle(cviai_frservice_handle_t *handle, cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    syslog(LOG_ERR, "ai_handle is empty.");
    return CVI_FAILURE;
  }
  cviai_frservice_context_t *ctx = new cviai_frservice_context_t;
  memset(&ctx->feature_array_ext.feature_array, 0, sizeof(cvai_service_feature_array_t));
  ctx->ai_handle = ai_handle;
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_DestroyHandle(cviai_frservice_handle_t handle) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  FreeFeatureArrayExt(&ctx->feature_array_ext);
  delete ctx->m_dt;
  delete ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_FRService_RegisterFeatureArray(cviai_frservice_handle_t handle,
                                              const cvai_service_feature_array_t featureArray) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  return RegisterFeatureArray(featureArray, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_FRService_FaceInfoMatching(cviai_frservice_handle_t handle,
                                          const cvai_face_info_t *face_info, const uint32_t k,
                                          uint32_t **index) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  return FeatureMatchingRaw((uint8_t *)face_info->face_feature.ptr, face_info->face_feature.type, k,
                            index, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_FRService_RawMatching(cviai_frservice_handle_t handle, const uint8_t *feature,
                                     const feature_type_e type, const uint32_t k,
                                     uint32_t **index) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  return FeatureMatchingRaw(feature, type, k, index, &ctx->feature_array_ext);
}

CVI_S32 CVI_AI_FRService_DigitalZoom(cviai_frservice_handle_t handle,
                                     const VIDEO_FRAME_INFO_S *inFrame, const cvai_face_t *meta,
                                     const float face_skip_ratio, const float trans_ratio,
                                     VIDEO_FRAME_INFO_S *outFrame) {
  cviai_frservice_context_t *ctx = static_cast<cviai_frservice_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, face_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_FRService_DrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame) {
  return cviai::service::DrawMeta(meta, frame);
}
