#include "tdl_utils.h"
#include <cstdlib>
#include <cstring>
int32_t CVI_TDL_InitObjectMeta(cvtdl_object_t *object_meta, int num_objects) {
  object_meta->info =
      (cvtdl_object_info_t *)malloc(num_objects * sizeof(cvtdl_object_info_t));
  memset(object_meta->info, 0, num_objects * sizeof(cvtdl_object_info_t));
  object_meta->size = num_objects;
  object_meta->width = 0;
  object_meta->height = 0;
  return 0;
}

int32_t CVI_TDL_ReleaseObjectMeta(cvtdl_object_t *object_meta) { return 0; }

int32_t CVI_TDL_InitFaceMeta(cvtdl_face_t *face_meta, int num_faces,
                             int num_landmark_per_face) {
  face_meta->info =
      (cvtdl_face_info_t *)malloc(num_faces * sizeof(cvtdl_face_info_t));
  memset(face_meta->info, 0, num_faces * sizeof(cvtdl_face_info_t));
  for (int i = 0; i < num_faces; i++) {
    face_meta->info[i].landmarks.x =
        (float *)malloc(num_landmark_per_face * sizeof(float));
    face_meta->info[i].landmarks.y =
        (float *)malloc(num_landmark_per_face * sizeof(float));
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

int32_t CVI_TDL_InitKeypointMeta(cvtdl_keypoint_t *keypoint_meta,
                                 int num_keypoints) {
  keypoint_meta->info = (cvtdl_keypoint_info_t *)malloc(
      num_keypoints * sizeof(cvtdl_keypoint_info_t));
  keypoint_meta->size = num_keypoints;
  keypoint_meta->width = 0;
  keypoint_meta->height = 0;
  return 0;
}

int32_t CVI_TDL_ReleaseKeypointMeta(cvtdl_keypoint_t *keypoint_meta) {
  free(keypoint_meta->info);
  return 0;
}
