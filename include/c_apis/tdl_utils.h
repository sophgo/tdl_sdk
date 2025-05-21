#ifndef TDL_UTILS_H
#define TDL_UTILS_H

#include "tdl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t TDL_InitObjectMeta(TDLObject *object_meta, int num_objects,
                           int num_landmark);

int32_t TDL_ReleaseObjectMeta(TDLObject *object_meta);

int32_t TDL_InitFaceMeta(TDLFace *face_meta, int num_faces,
                         int num_landmark_per_face);

int32_t TDL_ReleaseFaceMeta(TDLFace *face_meta);

int32_t TDL_InitSemanticSegMeta(TDLSegmentation *seg_meta, int output_size);

int32_t TDL_ReleaseSemanticSegMeta(TDLSegmentation *seg_meta);

int32_t TDL_InitInstanceSegMeta(TDLInstanceSeg *inst_seg_meta, int num_objects,
                                uint32_t mask_size);

int32_t TDL_ReleaseInstanceSegMeta(TDLInstanceSeg *inst_seg_meta);

int32_t TDL_InitKeypointMeta(TDLKeypoint *keypoint_meta, int num_keypoints);

int32_t TDL_ReleaseKeypointMeta(TDLKeypoint *keypoint_meta);

int32_t TDL_InitFeatureMeta(TDLFeature *feature_meta);

int32_t TDL_ReleaseFeatureMeta(TDLFeature *feature_meta);

int32_t TDL_RegisterFeature(const TDLFeature feature);

int32_t TDL_MatchFeature(const TDLFeature feature, TDLFeature *matched_feature);

int32_t TDL_CaculateSimilarity(const TDLFeature feature1,
                               const TDLFeature feature2, float *similarity);

int32_t TDL_InitLaneMeta(TDLLane *lane_meta, int output_size);

int32_t TDL_ReleaseLaneMeta(TDLLane *lane_meta);

int32_t TDL_InitCharacterMeta(TDLOcr *char_meta, int length);

int32_t TDL_ReleaseCharacterMeta(TDLOcr *char_meta);

int32_t TDL_InitTrackMeta(TDLTracker *track_meta, int num_track);

int32_t TDL_ReleaseTrackMeta(TDLTracker *track_meta);

#ifdef __cplusplus
}
#endif

#endif