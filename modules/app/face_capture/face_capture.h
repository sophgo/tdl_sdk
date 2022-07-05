#ifndef _CVIAI_APP_FACE_CAPTURE_H_
#define _CVIAI_APP_FACE_CAPTURE_H_

#include "app/capture/face_capture_type.h"
#include "core/cviai_core.h"

CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info);

CVI_S32 _FaceCapture_Init(face_capture_t **face_cpt_info, uint32_t buffer_size);

CVI_S32 _FaceCapture_QuickSetUp(cviai_handle_t ai_handle, face_capture_t *face_cpt_info,
                                int fd_model_id, int fr_model_id, const char *fd_model_path,
                                const char *fr_model_path, const char *fq_model_path);

CVI_S32 _FaceCapture_GetDefaultConfig(face_capture_config_t *cfg);

CVI_S32 _FaceCapture_SetConfig(face_capture_t *face_cpt_info, face_capture_config_t *cfg,
                               cviai_handle_t ai_handle);

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         VIDEO_FRAME_INFO_S *frame);

CVI_S32 _FaceCapture_SetMode(face_capture_t *face_cpt_info, capture_mode_e mode);

CVI_S32 _FaceCapture_CleanAll(face_capture_t *face_cpt_info);

#endif  // End of _CVIAI_APP_FACE_CAPTURE_H_
