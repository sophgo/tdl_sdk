#include "tdl_utils.h"
#include <cstdlib>

int32_t CVI_TDL_InitObjectMeta(cvtdl_object_t *object_meta, int num_objects) {
  return 0;
}

int32_t CVI_TDL_ReleaseObjectMeta(cvtdl_object_t *object_meta) { return 0; }

int32_t CVI_TDL_InitFaceMeta(cvtdl_face_t *face_meta, int num_faces,
                             int num_landmark_per_face) {
  face_meta->info = (cvtdl_face_info_t *)malloc(num_faces * sizeof(cvtdl_face_info_t));
  for (int i = 0 ;i < num_faces; i ++) {
    face_meta->info[i].landmarks.x = (float *)malloc(num_landmark_per_face * sizeof(float));
    face_meta->info[i].landmarks.y = (float *)malloc(num_landmark_per_face * sizeof(float));
  }
  return 0;
}

int32_t CVI_TDL_ReleaseFaceMeta(cvtdl_face_t *face_meta) {
  for (int i = 0; i < face_meta->size; i++) {
    free(face_meta->info[i].landmarks.x);
    free(face_meta->info[i].landmarks.y);
  }
  free(face_meta->info);
  return 0;
}

int32_t CVI_TDL_InitClassInfoMeta(cvtdl_class_info_t *class_info,
                                  int num_classes) {
  return 0;
}

int32_t CVI_TDL_ReleaseClassInfoMeta(cvtdl_class_info_t *class_info) {
  return 0;
}
