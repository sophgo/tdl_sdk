#ifndef TDL_UTILS_H
#define TDL_UTILS_H

#include "tdl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t TDL_InitObjectMeta(tdl_object_t *object_meta,
                           int num_objects,
                           int num_landmark);

int32_t TDL_ReleaseObjectMeta(tdl_object_t *object_meta);

int32_t TDL_InitFaceMeta(tdl_face_t *face_meta,
                         int num_faces,
                         int num_landmark_per_face);

int32_t TDL_ReleaseFaceMeta(tdl_face_t *face_meta);

int32_t TDL_InitSemanticSegMeta(tdl_seg_t *seg_meta,
                                int output_size);

int32_t TDL_ReleaseSemanticSegMeta(tdl_seg_t *seg_meta);

int32_t TDL_InitInstanceSegMeta(tdl_instance_seg_t *inst_seg_meta,
                                int num_objects,
                                uint32_t mask_size);

int32_t TDL_ReleaseInstanceSegMeta(tdl_instance_seg_t *inst_seg_meta);

int32_t TDL_InitKeypointMeta(tdl_keypoint_t *keypoint_meta,
                             int num_keypoints);

int32_t TDL_ReleaseKeypointMeta(tdl_keypoint_t *keypoint_meta);

int32_t TDL_InitFeatureMeta(tdl_feature_t *feature_meta);

int32_t TDL_ReleaseFeatureMeta(tdl_feature_t *feature_meta);

int32_t TDL_RegisterFeature(const tdl_feature_t feature);

int32_t TDL_MatchFeature(const tdl_feature_t feature,
                         tdl_feature_t *matched_feature);

int32_t TDL_CaculateSimilarity(const tdl_feature_t feature1,
                               const tdl_feature_t feature2,
                               float *similarity);

int32_t TDL_InitLaneMeta(tdl_lane_t *lane_meta,
                         int output_size);

int32_t TDL_ReleaseLaneMeta(tdl_lane_t *lane_meta);

int32_t TDL_InitCharacterMeta(tdl_ocr_t *char_meta, int length);

int32_t TDL_ReleaseCharacterMeta(tdl_ocr_t *char_meta);

#ifdef __cplusplus
}
#endif

#endif