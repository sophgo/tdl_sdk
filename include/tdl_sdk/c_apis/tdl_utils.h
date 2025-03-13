#ifndef TDL_UTILS_H
#define TDL_UTILS_H

#include "tdl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t CVI_TDL_InitObjectMeta(cvtdl_object_t *object_meta, int num_objects);

int32_t CVI_TDL_ReleaseObjectMeta(cvtdl_object_t *object_meta);

int32_t CVI_TDL_InitFaceMeta(cvtdl_face_t *face_meta, int num_faces,
                             int num_landmark_per_face);

int32_t CVI_TDL_ReleaseFaceMeta(cvtdl_face_t *face_meta);

int32_t CVI_TDL_InitKeypointMeta(cvtdl_keypoint_t *keypoint_meta,
                                 int num_keypoints);

int32_t CVI_TDL_ReleaseKeypointMeta(cvtdl_keypoint_t *keypoint_meta);

#ifdef __cplusplus
}
#endif

#endif