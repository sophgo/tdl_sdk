#ifndef _CVIAI_APP_FACE_CAPTURE_H_
#define _CVIAI_APP_FACE_CAPTURE_H_

#include "core/cviai_core.h"
// #include "core/cvai_core_types.h"
// #include "face/cvai_face_types.h"
#include "app/face_capture/face_capture_type.h"

CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info);

CVI_S32 _FaceCapture_Init(face_capture_t **face_cpt_info);

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         VIDEO_FRAME_INFO_S *frame);

#endif  // End of _CVIAI_APP_FACE_CAPTURE_H_