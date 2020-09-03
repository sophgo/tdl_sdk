#include "objservice/cviai_objservice.h"

#include "cviai_core_internal.hpp"
#include "digital_tracking/digital_tracking.hpp"
#include "draw_rect/draw_rect.hpp"

#include <cvimath/cvimath.h>

typedef struct {
  cviai_handle_t ai_handle = NULL;
  cviai::service::DigitalTracking *m_dt = nullptr;
} cviai_objservice_context_t;

CVI_S32 CVI_AI_OBJService_CreateHandle(cviai_objservice_handle_t *handle,
                                       cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
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
  delete ctx;
  return CVI_SUCCESS;
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