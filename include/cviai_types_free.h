#ifndef _CVIAI_TYPES_FREE_H_
#define _CVIAI_TYPES_FREE_H_
#include "core/cvi_core_types.h"
#include "face/cvi_face_types.h"
#include "object/cvi_object_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void CVI_AI_FreeFeature(cvi_feature_t *feature);
void CVI_AI_FreePts(cvi_pts_t *pts);

void CVI_AI_FreeFaceInfo(cvi_face_info_t *face_info);
void CVI_AI_FreeFace(cvi_face_t *face);

void CVI_AI_FreeObjectInfo(cvi_object_info_t *face_info);
void CVI_AI_FreeObject(cvi_object_t *face);

#ifdef __cplusplus
}
#endif
#endif