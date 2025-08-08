#ifndef TDL_TYPES_H
#define TDL_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include "tdl_object_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TDLHandle;
typedef void *TDLHandleEx;
typedef void *TDLImage;

typedef enum {
  TDL_TYPE_INT8 = 0, /**< Equals to int8_t. */
  TDL_TYPE_UINT8,    /**< Equals to uint8_t. */
  TDL_TYPE_INT16,    /**< Equals to int16_t. */
  TDL_TYPE_UINT16,   /**< Equals to uint16_t. */
  TDL_TYPE_INT32,    /**< Equals to int32_t. */
  TDL_TYPE_UINT32,   /**< Equals to uint32_t. */
  TDL_TYPE_BF16,     /**< Equals to bf16. */
  TDL_TYPE_FP16,     /**< Equals to fp16. */
  TDL_TYPE_FP32,     /**< Equals to fp32. */
  TDL_TYPE_UNKOWN    /**< Equals to unkown. */
} TDLDataTypeE;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
} TDLBox;

typedef struct {
  int8_t *ptr;
  uint32_t size;
  TDLDataTypeE type;
} TDLFeature;

typedef struct {
  uint32_t size;
  TDLFeature *feature;
} TDLFeatureInfo;

typedef struct {
  float *x;
  float *y;
  uint32_t size;
  float score;
} TDLPoints;

typedef struct {
  float x;
  float y;
  float score;
} TDLLandmarkInfo;

typedef struct {
  char name[128];
  TDLBox box;
  float score;
  int class_id;
  uint64_t track_id;
  uint32_t landmark_size;
  TDLLandmarkInfo *landmark_properity;
  TDLObjectTypeE obj_type;
} TDLObjectInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  TDLObjectInfo *info;
} TDLObject;

typedef struct {
  char name[128];
  float score;
  uint64_t track_id;
  TDLBox box;
  TDLPoints landmarks;
  TDLFeature feature;

  float gender_score;
  float glass_score;
  float age;
  float liveness_score;
  float hardhat_score;
  float mask_score;

  float recog_score;
  float face_quality;
  float pose_score;
  float blurness;
} TDLFaceInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  TDLFaceInfo *info;
} TDLFace;

typedef struct {
  int32_t class_id;
  float score;
} TDLClassInfo;

typedef struct {
  uint32_t size;

  TDLClassInfo *info;
} TDLClass;

typedef struct {
  float x;
  float y;
  float score;
} TDLKeypointInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  TDLKeypointInfo *info;
} TDLKeypoint;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t output_width;
  uint32_t output_height;
  uint8_t *class_id;
  uint8_t *class_conf;
} TDLSegmentation;

typedef struct {
  uint8_t *mask;
  float *mask_point;
  uint32_t mask_point_size;
  TDLObjectInfo *obj_info;
} TDLInstanceSegInfo;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  uint32_t mask_width;
  uint32_t mask_height;
  TDLInstanceSegInfo *info;
} TDLInstanceSeg;

typedef struct {
  float x[2];
  float y[2];
  float score;
} TDLLanePoint;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  TDLLanePoint *lane;
  int lane_state;
} TDLLane;

typedef struct {
  int w;
  int h;
  int8_t *int_logits;
} TDLDepthLogits;

typedef struct {
  uint64_t id;
  TDLBox bbox;
} TDLTrackerInfo;
typedef struct {
  uint32_t size;
  int out_num;
  TDLTrackerInfo *info;
} TDLTracker;

typedef struct {
  uint32_t size;
  char *text_info;
} TDLOcr;

typedef struct {
  float quality;
  uint64_t snapshot_frame_id;
  uint64_t track_id;
  bool male;
  bool glass;
  uint8_t age;
  uint8_t emotion;
  TDLImage object_image;
} TDLSnapshotInfo;

typedef struct {
  uint32_t snapshot_size;
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  TDLFace face_meta;
  TDLObject person_meta;
  TDLObject pet_meta;
  TDLTracker track_meta;
  TDLSnapshotInfo *snapshot_info;
  TDLFeature *features;
  TDLImage image;
} TDLCaptureInfo;

typedef struct {
  float awb[3];  // rgain, ggain, bgain
  float ccm[9];  // rgb[3][3]
  float blc;
} TDLIspMeta;

typedef struct {
  int r;
  int g;
  int b;
} color_rgb;

typedef struct {
  color_rgb color;
  uint32_t size;
} TDLBrush;

#ifdef __cplusplus
}
#endif
#endif
