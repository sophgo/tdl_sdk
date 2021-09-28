#include "app/cviai_app.h"
#include "cviai_log.hpp"

#include "face_capture/face_capture.h"
// #include "face_management/face_management.h"

CVI_S32 CVI_AI_APP_CreateHandle(cviai_app_handle_t *handle, cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    LOGE("ai_handle is empty.");
    return CVIAI_FAILURE;
  }
  cviai_app_context_t *ctx = (cviai_app_context_t *)malloc(sizeof(cviai_app_context_t));
  ctx->ai_handle = ai_handle;
  ctx->face_cpt_info = NULL;
  // ctx->face_mng_info = NULL;
  *handle = ctx;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_DestroyHandle(cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  _FaceCapture_Free(ctx->face_cpt_info);
  ctx->face_cpt_info = NULL;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceCapture_Init(const cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  _FaceCapture_Init(&(ctx->face_cpt_info));
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceCapture_Run(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S *frame) {
  cviai_app_context_t *ctx = handle;
  _FaceCapture_Run(ctx->face_cpt_info, ctx->ai_handle, frame);
  return CVIAI_SUCCESS;
}

#if 0
CVI_S32 CVI_AI_APP_FaceManagement_Init(const cviai_app_handle_t handle){
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceManagement_Run(const cviai_app_handle_t handle,
                                      VIDEO_FRAME_INFO_S *frame){
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceManagement_Login(const cviai_app_handle_t handle, const cvai_face_t *faces){
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceManagement_Search(const cviai_app_handle_t handle, const cvai_face_t *faces){
  return CVIAI_SUCCESS;
}
#endif