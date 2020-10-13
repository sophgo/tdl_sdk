#include "service/cviai_objservice.h"

#include "area_detect/area_detect.hpp"
#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"
#include "feature_matching/feature_matching.hpp"

#include <cvimath/cvimath.h>

typedef struct {
  cvai_service_feature_matching_e matching_method;
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
                                               const cvai_service_feature_array_t featureArray,
                                               const cvai_service_feature_matching_e method) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  ctx->matching_method = method;
  if (ctx->matching_method == cvai_service_feature_matching_e::INNER_PRODUCT) {
    return RegisterIPFeatureArray(featureArray, &ctx->feature_array_ext);
  }
  LOGE("Unsupported method %u\n", ctx->matching_method);
  return CVI_FAILURE;
}

CVI_S32 CVI_AI_OBJService_ObjectInfoMatching(cviai_objservice_handle_t handle,
                                             const cvai_object_info_t *object_info,
                                             const uint32_t k, uint32_t **index) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->matching_method == cvai_service_feature_matching_e::INNER_PRODUCT) {
    return FeatureMatchingIPRaw((uint8_t *)object_info->feature.ptr, object_info->feature.type, k,
                                index, &ctx->feature_array_ext);
  }
  LOGE("Unsupported method %u\n", ctx->matching_method);
  return CVI_FAILURE;
}

CVI_S32 CVI_AI_OBJService_RawMatching(cviai_objservice_handle_t handle, const uint8_t *feature,
                                      const feature_type_e type, const uint32_t k,
                                      uint32_t **index) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->matching_method == cvai_service_feature_matching_e::INNER_PRODUCT) {
    return FeatureMatchingIPRaw(feature, type, k, index, &ctx->feature_array_ext);
  }
  LOGE("Unsupported method %u\n", ctx->matching_method);
  return CVI_FAILURE;
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

CVI_S32 CVI_AI_OBJService_SetIntersect(cviai_objservice_handle_t handle, const cvai_pts_t *pts) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->m_ad == nullptr) {
    ctx->m_ad = new cviai::service::AreaDetect();
  }
  return ctx->m_ad->setArea(*pts);
}

CVI_S32 CVI_AI_OBJService_DetectIntersect(cviai_objservice_handle_t handle,
                                          const VIDEO_FRAME_INFO_S *frame,
                                          const cvai_object_t *obj_meta,
                                          cvai_area_detect_e **status) {
  cviai_objservice_context_t *ctx = static_cast<cviai_objservice_context_t *>(handle);
  if (ctx->m_ad == nullptr) {
    LOGE("Please set intersect area first.\n");
    return CVI_FAILURE;
  }
  int ret = CVI_SUCCESS;
  if (*status != NULL) {
    free(*status);
  }
  *status = (cvai_area_detect_e *)malloc(obj_meta->size * sizeof(cvai_area_detect_e));
  for (uint32_t i = 0; i < obj_meta->size; ++i) {
    float &&center_pts_x = (obj_meta->info[i].bbox.x1 + obj_meta->info[i].bbox.x2) / 2;
    float &&center_pts_y = (obj_meta->info[i].bbox.y1 + obj_meta->info[i].bbox.y2) / 2;
    ctx->m_ad->run(frame->stVFrame.u64PTS, obj_meta->info[i].unique_id, center_pts_x, center_pts_y,
                   &(*status)[i]);
  }
  return ret;
}