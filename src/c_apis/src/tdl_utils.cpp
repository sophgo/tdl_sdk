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

int32_t CVI_TDL_InitInstanceSegMeta(cvtdl_object_t *object_meta, int num_objects, uint32_t mask_size) {

    object_meta->info = (cvtdl_object_info_t *)malloc(num_objects * sizeof(cvtdl_object_info_t));

    memset(object_meta->info, 0, num_objects * sizeof(cvtdl_object_info_t));
    
    for (int i = 0; i < num_objects; i++) {
        object_meta->info[i].mask_properity = (cvtdl_mask_info_t *)malloc(sizeof(cvtdl_mask_info_t));
        object_meta->info[i].mask_properity->mask = (uint8_t *)malloc(mask_size * sizeof(uint8_t)); 
        object_meta->info[i].mask_properity->mask_point = NULL; 
        object_meta->info[i].mask_properity->mask_point_size = 0; 
    }

    object_meta->size = num_objects;
    object_meta->width = 0;
    object_meta->height = 0;
    object_meta->mask_width = 0;
    object_meta->mask_height = 0;
    return 0;
}

int32_t CVI_TDL_ReleaseInstanceSegMeta(cvtdl_object_t *object_meta) { 
  for (int i = 0; i < object_meta->size; i++) {
    if (object_meta->info[i].mask_properity) {
      if (object_meta->info[i].mask_properity->mask) {
        free(object_meta->info[i].mask_properity->mask);
        object_meta->info[i].mask_properity->mask = NULL;
      }
      if (object_meta->info[i].mask_properity->mask_point) {
        free(object_meta->info[i].mask_properity->mask_point);
        object_meta->info[i].mask_properity->mask_point = NULL;
      }
      free(object_meta->info[i].mask_properity);
      object_meta->info[i].mask_properity = NULL;
    }
  }
  free(object_meta->info);
  return 0; 
}

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

int32_t CVI_TDL_InitSemanticSegMeta(cvtdl_seg_t *seg_meta, int output_size){
    seg_meta->class_id = (uint8_t *)malloc(output_size * sizeof(uint8_t));
    seg_meta->class_conf = (uint8_t *)malloc(output_size * sizeof(uint8_t));
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
 lane_meta->lane = (cvtdl_lane_point_t *)malloc(
      output_size * sizeof(cvtdl_lane_point_t));
  memset(lane_meta->lane, 0, output_size * sizeof(cvtdl_lane_point_t));      
}

int32_t CVI_TDL_ReleaseLaneMeta(cvtdl_lane_t *lane_meta) {
  if (lane_meta->lane != NULL) {
    free(lane_meta->lane);
    lane_meta->lane = NULL;
  }
  return 0;
}