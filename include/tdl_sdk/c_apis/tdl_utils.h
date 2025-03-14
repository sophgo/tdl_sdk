#ifndef TDL_UTILS_H
#define TDL_UTILS_H

#include "tdl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t CVI_TDL_InitObjectMeta(cvtdl_object_t *object_meta,
                               int num_objects,
                               int num_landmark);

int32_t CVI_TDL_ReleaseObjectMeta(cvtdl_object_t *object_meta);

int32_t CVI_TDL_InitFaceMeta(cvtdl_face_t *face_meta,
                             int num_faces,
                             int num_landmark_per_face);

int32_t CVI_TDL_ReleaseFaceMeta(cvtdl_face_t *face_meta);

int32_t CVI_TDL_InitSemanticSegMeta(cvtdl_seg_t *seg_meta,
                                    int output_size);

int32_t CVI_TDL_ReleaseSemanticSegMeta(cvtdl_seg_t *seg_meta);

int32_t CVI_TDL_InitInstanceSegMeta(cvtdl_instance_seg_t *inst_seg_meta,
                                    int num_objects,
                                    uint32_t mask_size);

int32_t CVI_TDL_ReleaseInstanceSegMeta(cvtdl_instance_seg_t *inst_seg_meta);

int32_t CVI_TDL_InitKeypointMeta(cvtdl_keypoint_t *keypoint_meta,
                                 int num_keypoints);

int32_t CVI_TDL_ReleaseKeypointMeta(cvtdl_keypoint_t *keypoint_meta);

int32_t CVI_TDL_InitFeatureMeta(cvtdl_feature_t *feature_meta);

int32_t CVI_TDL_ReleaseFeatureMeta(cvtdl_feature_t *feature_meta);

int32_t CVI_TDL_RegisterFeature(const cvtdl_feature_t feature);

int32_t CVI_TDL_MatchFeature(const cvtdl_feature_t feature,
                             cvtdl_feature_t *matched_feature);

int32_t CVI_TDL_CaculateSimilarity(const cvtdl_feature_t feature1,
                                   const cvtdl_feature_t feature2,
                                   float *similarity);
int32_t CVI_TDL_InitLaneMeta(cvtdl_lane_t *lane_meta, int output_size);

int32_t CVI_TDL_ReleaseLaneMeta(cvtdl_lane_t *lane_meta);

#ifdef __cplusplus
}
#endif

#endif