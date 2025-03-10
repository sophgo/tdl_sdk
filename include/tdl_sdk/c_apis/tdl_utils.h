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
int32_t CVI_TDL_InitClassInfoMeta(cvtdl_class_info_t *class_info,
                                  int num_classes);
int32_t CVI_TDL_ReleaseClassInfoMeta(cvtdl_class_info_t *class_info);
#ifdef __cplusplus
}
#endif

#endif