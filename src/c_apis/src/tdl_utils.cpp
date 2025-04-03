#include "tdl_utils.h"
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <stdio.h>
int32_t TDL_InitObjectMeta(TDLObject *object_meta, int num_objects, int num_landmark) {
  if (object_meta->info == NULL) {
    object_meta->info =
      (TDLObjectInfo *)malloc(num_objects * sizeof(TDLObjectInfo));
      for (int i = 0; i < num_objects; i++) {
        object_meta->info[i].landmark_properity = NULL;
      }
  }

  for (int i = 0; i < num_objects; i ++) {
    if (num_landmark > 0 && object_meta->info[i].landmark_properity == NULL) {
      object_meta->info[i].landmark_properity = (TDLLandmarkInfo *)malloc(
          num_landmark * sizeof(TDLLandmarkInfo));
    }
  }
  object_meta->size = num_objects;
  object_meta->width = 0;
  object_meta->height = 0;
  return 0;
}

int32_t TDL_ReleaseObjectMeta(TDLObject *object_meta) {
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

int32_t TDL_InitInstanceSegMeta(TDLInstanceSeg *inst_seg_meta, int num_objects, uint32_t mask_size) {
    if (inst_seg_meta->info != NULL) return 0;
    inst_seg_meta->info = (TDLInstanceSegInfo *)malloc(num_objects * sizeof(TDLInstanceSegInfo));

    memset(inst_seg_meta->info, 0, num_objects * sizeof(TDLInstanceSegInfo));

    for (int i = 0; i < num_objects; i++) {
      inst_seg_meta->info[i].mask = NULL;
      inst_seg_meta->info[i].mask_point = NULL;
      inst_seg_meta->info[i].mask_point_size = 0;
      inst_seg_meta->info[i].obj_info = (TDLObjectInfo *)malloc(sizeof(TDLObjectInfo));
    }

    inst_seg_meta->size = num_objects;
    inst_seg_meta->width = 0;
    inst_seg_meta->height = 0;
    inst_seg_meta->mask_width = 0;
    inst_seg_meta->mask_height = 0;
    return 0;
}

int32_t TDL_ReleaseInstanceSegMeta(TDLInstanceSeg *inst_seg_meta) { 
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

int32_t TDL_InitFaceMeta(TDLFace *face_meta, int num_faces,
                             int num_landmark_per_face) {
  if (face_meta->info != NULL) return 0;
  face_meta->info =
      (TDLFaceInfo *)malloc(num_faces * sizeof(TDLFaceInfo));
  memset(face_meta->info, 0, num_faces * sizeof(TDLFaceInfo));
  for (int i = 0; i < num_faces; i++) {
    face_meta->info[i].landmarks.x =
        (float *)malloc(num_landmark_per_face * sizeof(float));
    face_meta->info[i].landmarks.y =
        (float *)malloc(num_landmark_per_face * sizeof(float));
  }
  face_meta->size = num_faces;
  return 0;
}

int32_t TDL_ReleaseFaceMeta(TDLFace *face_meta) {
  for (int i = 0; i < face_meta->size; i++) {
    free(face_meta->info[i].landmarks.x);
    free(face_meta->info[i].landmarks.y);
  }
  free(face_meta->info);
  return 0;
}

int32_t TDL_InitKeypointMeta(TDLKeypoint *keypoint_meta,
                                 int num_keypoints) {
  if (keypoint_meta->info) return 0;
  keypoint_meta->info = (TDLKeypointInfo *)malloc(
      num_keypoints * sizeof(TDLKeypointInfo));
  keypoint_meta->size = num_keypoints;
  keypoint_meta->width = 0;
  keypoint_meta->height = 0;
  return 0;
}

int32_t TDL_ReleaseKeypointMeta(TDLKeypoint *keypoint_meta) {
  free(keypoint_meta->info);
  return 0;
}

int32_t TDL_InitSemanticSegMeta(TDLSegmentation *seg_meta, int output_size){
    seg_meta->class_id = NULL;
    seg_meta->class_conf = NULL;
    seg_meta->width = 0;
    seg_meta->height = 0;
    seg_meta->output_width = 0;
    seg_meta->output_height = 0;
    return 0;
}

int32_t TDL_ReleaseSemanticSegMeta(TDLSegmentation *seg_meta) {
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

int32_t TDL_InitLaneMeta(TDLLane *lane_meta, int output_size){
  if (lane_meta->lane) return 0;
  lane_meta->lane = (TDLLanePoint *)malloc(
      output_size * sizeof(TDLLanePoint));
  memset(lane_meta->lane, 0, output_size * sizeof(TDLLanePoint));

  lane_meta->size = output_size;
  return 0;
}

int32_t TDL_ReleaseLaneMeta(TDLLane *lane_meta) {
  if (lane_meta->lane != NULL) {
    free(lane_meta->lane);
    lane_meta->lane = NULL;
  }
  return 0;
}

int32_t TDL_InitCharacterMeta(TDLOcr *char_meta, int length){
  if (char_meta->text_info) return 0;
  char_meta->text_info = (char *)malloc(length * sizeof(char));
  memset(char_meta->text_info, 0, length * sizeof(char));
  char_meta->size = length;
  return 0;
};

int32_t TDL_ReleaseCharacterMeta(TDLOcr *char_meta){
  if (char_meta->text_info != NULL) {
    free(char_meta->text_info);
    char_meta->text_info = NULL;
  }
  return 0;
};

int32_t TDL_InitFeatureMeta(TDLFeature *feature_meta) {
  if (feature_meta->ptr != NULL) return 0;
  feature_meta->ptr = (int8_t *)malloc(sizeof(int8_t));
  return 0;
}

int32_t TDL_ReleaseFeatureMeta(TDLFeature *feature_meta) {
  if (feature_meta->ptr){
    free(feature_meta->ptr);
    feature_meta->ptr = NULL;
  }
  return 0;
}

int32_t TDL_CaculateSimilarity(const TDLFeature feature1,
                                   const TDLFeature feature2,
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