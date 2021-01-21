#include "service/cviai_service.h"

#include "area_detect/area_detect.hpp"
#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"
#include "face_angle/face_angle.hpp"
#include "feature_matching/feature_matching.hpp"

#include <cvimath/cvimath.h>

typedef struct {
  cviai_handle_t ai_handle = NULL;
  cviai::service::FeatureMatching *m_fm = nullptr;
  cviai::service::DigitalTracking *m_dt = nullptr;
  cviai::service::AreaDetect *m_ad = nullptr;
} cviai_service_context_t;

CVI_S32 CVI_AI_Service_CreateHandle(cviai_service_handle_t *handle, cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    LOGC("ai_handle is empty.");
    return CVI_FAILURE;
  }
  cviai_service_context_t *ctx = new cviai_service_context_t;
  ctx->ai_handle = ai_handle;
  *handle = ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Service_DestroyHandle(cviai_service_handle_t handle) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  delete ctx->m_fm;
  delete ctx->m_dt;
  delete ctx->m_ad;
  delete ctx;
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Service_RegisterFeatureArray(cviai_service_handle_t handle,
                                            const cvai_service_feature_array_t featureArray,
                                            const cvai_service_feature_matching_e method) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  int ret = CVI_SUCCESS;
  if (ctx->m_fm == nullptr) {
    ctx->m_fm = new cviai::service::FeatureMatching();
    if ((ret = ctx->m_fm->init()) != CVI_SUCCESS) {
      LOGE("Feature matching instance initialization failed with %#x!\n", ret);
      delete ctx->m_fm;
      ctx->m_fm = nullptr;
      return ret;
    }
  }
  return ctx->m_fm->registerData(featureArray, method);
}

CVI_S32 CVI_AI_Service_CalculateSimilarity(cviai_service_handle_t handle,
                                           const cvai_feature_t *feature_rhs,
                                           const cvai_feature_t *feature_lhs, float *score) {
  if (feature_lhs->type != feature_rhs->type) {
    LOGE("feature type not matched! rhs=%d, lhs=%d\n", feature_rhs->type, feature_lhs->type);
    return CVI_FAILURE;
  }

  if (feature_lhs->size != feature_rhs->size) {
    LOGE("feature size not matched!, rhs: %u, lhs: %u\n", feature_rhs->size, feature_lhs->size);
    return CVI_FAILURE;
  }

  if (feature_rhs->type == TYPE_INT8) {
    int32_t value1 = 0, value2 = 0, value3 = 0;
    for (uint32_t i = 0; i < feature_rhs->size; i++) {
      value1 += (short)feature_rhs->ptr[i] * feature_rhs->ptr[i];
      value2 += (short)feature_lhs->ptr[i] * feature_lhs->ptr[i];
      value3 += (short)feature_rhs->ptr[i] * feature_lhs->ptr[i];
    }

    *score = (float)value3 / (sqrt((double)value1) * sqrt((double)value2));
  } else {
    LOGE("Unsupported feature type: %d\n", feature_rhs->type);
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_Service_FaceInfoMatching(cviai_service_handle_t handle,
                                        const cvai_face_info_t *face_info, const uint32_t topk,
                                        float threshold, uint32_t *indices, float *sims,
                                        uint32_t *size) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  return ctx->m_fm->run((uint8_t *)face_info->feature.ptr, face_info->feature.type, topk, indices,
                        sims, size, threshold);
}

CVI_S32 CVI_AI_Service_ObjectInfoMatching(cviai_service_handle_t handle,
                                          const cvai_object_info_t *object_info,
                                          const uint32_t topk, float threshold, uint32_t *indices,
                                          float *sims, uint32_t *size) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  return ctx->m_fm->run((uint8_t *)object_info->feature.ptr, object_info->feature.type, topk,
                        indices, sims, size, threshold);
}

CVI_S32 CVI_AI_Service_RawMatching(cviai_service_handle_t handle, const void *feature,
                                   const feature_type_e type, const uint32_t topk, float threshold,
                                   uint32_t *indices, float *scores, uint32_t *size) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  return ctx->m_fm->run((uint8_t *)feature, type, topk, indices, scores, size, threshold);
}

CVI_S32 CVI_AI_Service_FaceDigitalZoom(cviai_service_handle_t handle,
                                       const VIDEO_FRAME_INFO_S *inFrame, const cvai_face_t *meta,
                                       const float face_skip_ratio, const float padding_ratio,
                                       const float trans_ratio, VIDEO_FRAME_INFO_S *outFrame) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, padding_ratio, padding_ratio, padding_ratio,
                        padding_ratio, face_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_Service_ObjectDigitalZoom(cviai_service_handle_t handle,
                                         const VIDEO_FRAME_INFO_S *inFrame,
                                         const cvai_object_t *meta, const float obj_skip_ratio,
                                         const float trans_ratio, const float padding_ratio,
                                         VIDEO_FRAME_INFO_S *outFrame) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, padding_ratio, padding_ratio, padding_ratio,
                        padding_ratio, obj_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_Service_ObjectDigitalZoomExt(cviai_service_handle_t handle,
                                            const VIDEO_FRAME_INFO_S *inFrame,
                                            const cvai_object_t *meta, const float obj_skip_ratio,
                                            const float trans_ratio, const float pad_ratio_left,
                                            const float pad_ratio_right, const float pad_ratio_top,
                                            const float pad_ratio_bottom,
                                            VIDEO_FRAME_INFO_S *outFrame) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  if (ctx->m_dt == nullptr) {
    ctx->m_dt = new cviai::service::DigitalTracking();
  }

  ctx->m_dt->setVpssEngine(CVI_AI_GetVpssEngine(ctx->ai_handle, 0));
  return ctx->m_dt->run(inFrame, meta, outFrame, pad_ratio_left, pad_ratio_right, pad_ratio_top,
                        pad_ratio_bottom, obj_skip_ratio, trans_ratio);
}

CVI_S32 CVI_AI_Service_FaceDrawRect(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame,
                                    const bool drawText) {
  return cviai::service::DrawMeta(meta, frame, drawText);
}

CVI_S32 CVI_AI_Service_ObjectDrawRect(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame,
                                      const bool drawText) {
  return cviai::service::DrawMeta(meta, frame, drawText);
}
CVI_S32 CVI_AI_Service_ObjectWriteText(char *name, int x, int y, VIDEO_FRAME_INFO_S *frame,
                                       const bool drawText) {
  return cviai::service::WriteMeta(name, x, y, frame, drawText);
}
CVI_S32 CVI_AI_Service_SetIntersect(cviai_service_handle_t handle, const cvai_pts_t *pts) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
  if (ctx->m_ad == nullptr) {
    ctx->m_ad = new cviai::service::AreaDetect();
  }
  return ctx->m_ad->setArea(*pts);
}

CVI_S32 CVI_AI_Service_ObjectDetectIntersect(cviai_service_handle_t handle,
                                             const VIDEO_FRAME_INFO_S *frame,
                                             const cvai_object_t *obj_meta,
                                             cvai_area_detect_e **status) {
  cviai_service_context_t *ctx = static_cast<cviai_service_context_t *>(handle);
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
CVI_S32 CVI_AI_Service_FaceAngle(const cvai_pts_t *pts, cvai_head_pose_t *hp) {
  return cviai::service::Predict(pts, hp);
}
CVI_S32 CVI_AI_Service_ObjectDrawPose(const cvai_object_t *meta, VIDEO_FRAME_INFO_S *frame) {
  return cviai::service::DrawPose17(meta, frame);
}
