#include "app/cviai_app.h"
#include "cviai_log.hpp"

#include "face_capture/face_capture.h"
#include "person_capture/person_capture.h"
#include "personvehicle_capture/personvehicle_capture.h"

CVI_S32 CVI_AI_APP_CreateHandle(cviai_app_handle_t *handle, cviai_handle_t ai_handle) {
  if (ai_handle == NULL) {
    LOGE("ai_handle is empty.");
    return CVIAI_FAILURE;
  }
  cviai_app_context_t *ctx = (cviai_app_context_t *)malloc(sizeof(cviai_app_context_t));
  ctx->ai_handle = ai_handle;
  ctx->face_cpt_info = NULL;
  ctx->person_cpt_info = NULL;
  ctx->personvehicle_cpt_info = NULL;
  *handle = ctx;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_APP_DestroyHandle(cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  _FaceCapture_Free(ctx->face_cpt_info);
  _PersonCapture_Free(ctx->person_cpt_info);
  _PersonVehicleCapture_Free(ctx->personvehicle_cpt_info);
  ctx->face_cpt_info = NULL;
  ctx->person_cpt_info = NULL;
  ctx->personvehicle_cpt_info = NULL;
  return CVIAI_SUCCESS;
}

/* Face Capture */
CVI_S32 CVI_AI_APP_FaceCapture_Init(const cviai_app_handle_t handle, uint32_t buffer_size) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_Init(&(ctx->face_cpt_info), buffer_size);
}

CVI_S32 CVI_AI_APP_FaceCapture_QuickSetUp(const cviai_app_handle_t handle, int fd_model_id,
                                          int fr_model_id, const char *fd_model_path,
                                          const char *fr_model_path, const char *fq_model_path,
                                          const char *fl_model_path) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_QuickSetUp(ctx->ai_handle, ctx->face_cpt_info, fd_model_id, fr_model_id,
                                 fd_model_path, fr_model_path, fq_model_path, fl_model_path);
}

CVI_S32 CVI_AI_APP_FaceCapture_FusePedSetup(const cviai_app_handle_t handle, int ped_model_id,
                                            const char *ped_model_path) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_FusePedSetUp(ctx->ai_handle, ctx->face_cpt_info, ped_model_id,
                                   ped_model_path);
}

CVI_S32 CVI_AI_APP_FaceCapture_GetDefaultConfig(face_capture_config_t *cfg) {
  return _FaceCapture_GetDefaultConfig(cfg);
}

CVI_S32 CVI_AI_APP_FaceCapture_SetConfig(const cviai_app_handle_t handle,
                                         face_capture_config_t *cfg) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_SetConfig(ctx->face_cpt_info, cfg, handle->ai_handle);
}

CVI_S32 CVI_AI_APP_FaceCapture_Run(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S *frame) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_Run(ctx->face_cpt_info, ctx->ai_handle, frame);
}

CVI_S32 CVI_AI_APP_FaceCapture_FDFR(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S *frame,
                                    cvai_face_t *p_face) {
  cviai_app_context_t *ctx = handle;
  face_capture_t *face_cpt_info = ctx->face_cpt_info;
  cviai_handle_t ai_handle = ctx->ai_handle;
  if (CVI_SUCCESS != face_cpt_info->fd_inference(ai_handle, frame, p_face)) {
    printf("fd_inference failed\n");
    return CVI_FAILURE;
  }
  LOGI("detect face num:%u\n", p_face->size);

  if (CVI_SUCCESS != face_cpt_info->fr_inference(ai_handle, frame, p_face)) {
    printf("fr inference failed\n");
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_APP_FaceCapture_SetMode(const cviai_app_handle_t handle, capture_mode_e mode) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_SetMode(ctx->face_cpt_info, mode);
}

CVI_S32 CVI_AI_APP_FaceCapture_CleanAll(const cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  return _FaceCapture_CleanAll(ctx->face_cpt_info);
}

/* Person Capture */
CVI_S32 CVI_AI_APP_PersonCapture_Init(const cviai_app_handle_t handle, uint32_t buffer_size) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_Init(&(ctx->person_cpt_info), buffer_size);
}

CVI_S32 CVI_AI_APP_PersonCapture_QuickSetUp(const cviai_app_handle_t handle,
                                            const char *od_model_name, const char *od_model_path,
                                            const char *reid_model_path) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_QuickSetUp(ctx->ai_handle, ctx->person_cpt_info, od_model_name,
                                   od_model_path, reid_model_path);
}

CVI_S32 CVI_AI_APP_PersonCapture_GetDefaultConfig(person_capture_config_t *cfg) {
  return _PersonCapture_GetDefaultConfig(cfg);
}

CVI_S32 CVI_AI_APP_PersonCapture_SetConfig(const cviai_app_handle_t handle,
                                           person_capture_config_t *cfg) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_SetConfig(ctx->person_cpt_info, cfg, handle->ai_handle);
}

CVI_S32 CVI_AI_APP_PersonCapture_Run(const cviai_app_handle_t handle, VIDEO_FRAME_INFO_S *frame) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_Run(ctx->person_cpt_info, ctx->ai_handle, frame);
}

// cousumer counting
CVI_S32 CVI_AI_APP_ConsumerCounting_Run(const cviai_app_handle_t handle,
                                        VIDEO_FRAME_INFO_S *frame) {
  cviai_app_context_t *ctx = handle;
  return _ConsumerCounting_Run(ctx->person_cpt_info, ctx->ai_handle, frame);
}
DLL_EXPORT CVI_S32 CVI_AI_APP_ConsumerCounting_Line(const cviai_app_handle_t handle, int A_x,
                                                    int A_y, int B_x, int B_y,
                                                    statistics_mode s_mode) {
  cviai_app_context_t *ctx = handle;
  return _ConsumerCounting_Line(ctx->person_cpt_info, A_x, A_y, B_x, B_y, s_mode);
}

CVI_S32 CVI_AI_APP_PersonCapture_SetMode(const cviai_app_handle_t handle, capture_mode_e mode) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_SetMode(ctx->person_cpt_info, mode);
}

CVI_S32 CVI_AI_APP_PersonCapture_CleanAll(const cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  return _PersonCapture_CleanAll(ctx->person_cpt_info);
}

// personvehicle cross the border
CVI_S32 CVI_AI_APP_PersonVehicleCapture_Init(const cviai_app_handle_t handle,
                                             uint32_t buffer_size) {
  cviai_app_context_t *ctx = handle;
  printf("2353536346346436\n");
  return _PersonVehicleCapture_Init(&(ctx->personvehicle_cpt_info), buffer_size);
}

CVI_S32 CVI_AI_APP_PersonVehicleCapture_QuickSetUp(const cviai_app_handle_t handle,
                                                   const char *od_model_name,
                                                   const char *od_model_path,
                                                   const char *reid_model_path) {
  cviai_app_context_t *ctx = handle;
  return _PersonVehicleCapture_QuickSetUp(ctx->ai_handle, ctx->personvehicle_cpt_info,
                                          od_model_name, od_model_path, reid_model_path);
}

CVI_S32 CVI_AI_APP_PersonVehicleCapture_GetDefaultConfig(personvehicle_capture_config_t *cfg) {
  return _PersonVehicleCapture_GetDefaultConfig(cfg);
}

CVI_S32 CVI_AI_APP_PersonVehicleCapture_SetConfig(const cviai_app_handle_t handle,
                                                  personvehicle_capture_config_t *cfg) {
  cviai_app_context_t *ctx = handle;
  return _PersonVehicleCapture_SetConfig(ctx->personvehicle_cpt_info, cfg, handle->ai_handle);
}

CVI_S32 CVI_AI_APP_PersonVehicleCapture_Run(const cviai_app_handle_t handle,
                                            VIDEO_FRAME_INFO_S *frame) {
  cviai_app_context_t *ctx = handle;
  return _PersonVehicleCapture_Run(ctx->personvehicle_cpt_info, ctx->ai_handle, frame);
}
DLL_EXPORT CVI_S32 CVI_AI_APP_PersonVehicleCapture_Line(const cviai_app_handle_t handle, int A_x,
                                                        int A_y, int B_x, int B_y,
                                                        statistics_mode s_mode) {
  cviai_app_context_t *ctx = handle;
  return _PersonVehicleCapture_Line(ctx->personvehicle_cpt_info, A_x, A_y, B_x, B_y, s_mode);
}

CVI_S32 CVI_AI_APP_PersonVehicleCapture_CleanAll(const cviai_app_handle_t handle) {
  cviai_app_context_t *ctx = handle;
  return _PersonVehicleCapture_CleanAll(ctx->personvehicle_cpt_info);
}
