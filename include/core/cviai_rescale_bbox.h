#ifndef _CVIAI_BBOX_RESCALE_H_
#define _CVIAI_BBOX_RESCALE_H_
#include "core/core/cvai_core_types.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_vb.h>
#include <cvi_sys.h>

#ifdef __cplusplus
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);
#endif

#ifdef __cplusplus
extern "C" {
#endif

DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxCenterFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxCenterObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxRBFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face);
DLL_EXPORT CVI_S32 CVI_AI_RescaleBBoxRBObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj);

#ifdef __cplusplus
}
#endif
#endif  // End of _CVIAI_BBOX_RESCALE_H_
