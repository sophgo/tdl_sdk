#ifndef _CVIAI_APP_FACE_CAPTURE_H_
#define _CVIAI_APP_FACE_CAPTURE_H_

#include "core/cviai_core.h"
// #include "core/cvai_core_types.h"
// #include "face/cvai_face_types.h"
#include "app/face_capture/face_capture_type.h"

CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle);

CVI_S32 _FaceCapture_Init(face_capture_t **face_cpt_info, uint32_t buffer_size);

CVI_S32 _FaceCapture_QuickSetUp(cviai_handle_t ai_handle, const char *fd_model_path,
                                const char *fq_model_path);

CVI_S32 _FaceCapture_GetDefaultConfig(face_capture_config_t *cfg);

CVI_S32 _FaceCapture_SetConfig(face_capture_t *face_cpt_info, face_capture_config_t *cfg);

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         const IVE_HANDLE ive_handle, VIDEO_FRAME_INFO_S *frame);

CVI_S32 _FaceCapture_SetMode(face_capture_t *face_cpt_info, capture_mode_e mode);

CVI_S32 _FaceCapture_CleanAll(face_capture_t *face_cpt_info);

#endif  // End of _CVIAI_APP_FACE_CAPTURE_H_