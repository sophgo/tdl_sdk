#include "tdl_utils.h"
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <stdio.h>
int32_t CVI_TDL_InitObjectMeta(cvtdl_object_t *object_meta, int num_objects, int num_landmark) {
  if (object_meta->info != NULL) return 0;
  object_meta->info =
      (cvtdl_object_info_t *)malloc(num_objects * sizeof(cvtdl_object_info_t));

  for (int i = 0; i < num_objects; i ++) {
    object_meta->info[i].landmark_properity = (cvtdl_landmark_info_t *)malloc(
      num_landmark * sizeof(cvtdl_landmark_info_t));
  }
  object_meta->size = num_objects;
  object_meta->width = 0;
  object_meta->height = 0;
  return 0;
}

int32_t CVI_TDL_ReleaseObjectMeta(cvtdl_object_t *object_meta) {
  for (int i = 0; i < object_meta->size; i ++) {
    if (object_meta->info[i].landmark_properity) {
      free(object_meta->info[i].landmark_properity);
      object_meta->info[i].landmark_properity = NULL;
    }
  }

  if (object_meta->info) {
    free(object_meta->info);
    object_meta->info = NULL;
  }

  return 0;
}

int32_t CVI_TDL_InitInstanceSegMeta(cvtdl_instance_seg_t *inst_seg_meta, int num_objects, uint32_t mask_size) {
    if (inst_seg_meta->info != NULL) return 0;
    inst_seg_meta->info = (cvtdl_instance_seg_info_t *)malloc(num_objects * sizeof(cvtdl_instance_seg_info_t));

    memset(inst_seg_meta->info, 0, num_objects * sizeof(cvtdl_instance_seg_info_t));

    for (int i = 0; i < num_objects; i++) {
      inst_seg_meta->info[i].mask = NULL;
      inst_seg_meta->info[i].mask_point = NULL;
      inst_seg_meta->info[i].mask_point_size = 0;
      inst_seg_meta->info[i].obj_info = (cvtdl_object_info_t *)malloc(sizeof(cvtdl_object_info_t));
    }

    inst_seg_meta->size = num_objects;
    inst_seg_meta->width = 0;
    inst_seg_meta->height = 0;
    inst_seg_meta->mask_width = 0;
    inst_seg_meta->mask_height = 0;
    return 0;
}

int32_t CVI_TDL_ReleaseInstanceSegMeta(cvtdl_instance_seg_t *inst_seg_meta) {
  for (int i = 0; i < inst_seg_meta->size; i++) {
    if (inst_seg_meta->info[i].obj_info) {
      free(inst_seg_meta->info[i].obj_info);
      inst_seg_meta->info[i].obj_info = NULL;
    }
    if (inst_seg_meta->info[i].mask) {
      free(inst_seg_meta->info[i].mask);
      inst_seg_meta->info[i].mask = NULL;
    }
    if (inst_seg_meta->info[i].mask_point) {
      free(inst_seg_meta->info[i].mask_point);
      inst_seg_meta->info[i].mask_point = NULL;
    }
  }

  free(inst_seg_meta->info);
  return 0;
}

int32_t CVI_TDL_InitFaceMeta(cvtdl_face_t *face_meta, int num_faces,
                             int num_landmark_per_face) {
  if (face_meta->info != NULL) return 0;
  face_meta->info =
      (cvtdl_face_info_t *)malloc(num_faces * sizeof(cvtdl_face_info_t));
  memset(face_meta->info, 0, num_faces * sizeof(cvtdl_face_info_t));
  for (int i = 0; i < num_faces; i++) {
    face_meta->info[i].landmarks.x =
        (float *)malloc(num_landmark_per_face * sizeof(float));
    face_meta->info[i].landmarks.y =
        (float *)malloc(num_landmark_per_face * sizeof(float));
  }
  face_meta->size = num_faces;
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
  if (keypoint_meta->info) return 0;
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

int32_t CVI_TDL_InitSemanticSegMeta(cvtdl_seg_t *seg_meta, int output_size){
  seg_meta->class_id = NULL;
  seg_meta->class_conf = NULL;
  seg_meta->width = 0;
  seg_meta->height = 0;
  seg_meta->output_width = 0;
  seg_meta->output_height = 0;
  return 0;
}

int32_t CVI_TDL_ReleaseSemanticSegMeta(cvtdl_seg_t *seg_meta) {
  if (seg_meta->class_id) {
    free(seg_meta->class_id);
    seg_meta->class_id = NULL;
  }
  if (seg_meta->class_conf) {
    free(seg_meta->class_conf);
    seg_meta->class_conf = NULL;
  }

  return 0;
}

int32_t CVI_TDL_InitLaneMeta(cvtdl_lane_t *lane_meta, int output_size){
  if (lane_meta->lane) return 0;
  lane_meta->lane = (cvtdl_lane_point_t *)malloc(
      output_size * sizeof(cvtdl_lane_point_t));
  memset(lane_meta->lane, 0, output_size * sizeof(cvtdl_lane_point_t));
  return 0;
}

int32_t CVI_TDL_ReleaseLaneMeta(cvtdl_lane_t *lane_meta) {
  if (lane_meta->lane != NULL) {
    free(lane_meta->lane);
    lane_meta->lane = NULL;
  }
  return 0;
}

int32_t CVI_TDL_InitFeatureMeta(cvtdl_feature_t *feature_meta) {
  if (feature_meta->ptr != NULL) return 0;
  feature_meta->ptr = (int8_t *)malloc(sizeof(int8_t));
  return 0;
}

int32_t CVI_TDL_ReleaseFeatureMeta(cvtdl_feature_t *feature_meta) {
  if (feature_meta->ptr){
    free(feature_meta->ptr);
    feature_meta->ptr = NULL;
  }
  return 0;
}

int32_t CVI_TDL_CaculateSimilarity(const cvtdl_feature_t feature1,
                                   const cvtdl_feature_t feature2,
                                   float *similarity) {
  *similarity = 0;
  if (feature1.size != feature2.size) {
    printf("feature1.size is not equal to feature2.size");
    return -1;
  }
  float norm1 = 0;
  float norm2 = 0;
  for (size_t i = 0; i < feature1.size; i++)
  {
    *similarity += feature1.ptr[i] * feature2.ptr[i];
    norm1 += feature1.ptr[i] * feature1.ptr[i];
    norm2 += feature2.ptr[i] * feature2.ptr[i];
  }
  norm1 = sqrt(norm1);
  norm2 = sqrt(norm2);
  *similarity = *similarity / (norm1 * norm2);
  return 0;
}