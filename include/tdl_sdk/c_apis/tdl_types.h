#ifndef TDL_TYPES_H
#define TDL_TYPES_H

#include <stdbool.h>
#include <stdint.h>
#include "tdl_object_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TDL_TYPE_INT8 = 0, /**< Equals to int8_t. */
  TDL_TYPE_UINT8,    /**< Equals to uint8_t. */
  TDL_TYPE_INT16,    /**< Equals to int16_t. */
  TDL_TYPE_UINT16,   /**< Equals to uint16_t. */
  TDL_TYPE_INT32,    /**< Equals to int32_t. */
  TDL_TYPE_UINT32,   /**< Equals to uint32_t. */
  TDL_TYPE_BF16,     /**< Equals to bf17. */
  TDL_TYPE_FP16,     /**< Equals to fp16. */
  TDL_TYPE_FP32,     /**< Equals to fp32. */
  TDL_TYPE_UNKOWN    /**< Equals to unkown. */
} cvtdl_data_type_e;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
} cvtdl_box_t;

typedef struct {
  int8_t *ptr;
  uint32_t size;
  cvtdl_data_type_e type;
} cvtdl_feature_t;

typedef struct {
  float *x;
  float *y;
  uint32_t size;
  float score;
} cvtdl_pts_t;

typedef struct {
  cvtdl_box_t box;
  float score;
  int class_id;
  cvtdl_object_type_e obj_type;
} cvtdl_object_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  cvtdl_object_info_t *info;
} cvtdl_object_t;

typedef struct {
  char name[128];
  float score;
  uint64_t track_id;
  cvtdl_box_t box;
  cvtdl_pts_t landmarks;
  cvtdl_feature_t feature;

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
} cvtdl_face_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvtdl_face_info_t *info;
} cvtdl_face_t;

typedef struct {
  int32_t class_id;
  float score;
} cvtdl_class_info_t;

typedef struct {
  uint32_t size;

  cvtdl_class_info_t *info;
} cvtdl_class_t;

typedef struct {
  float x;
  float y;
  float score;
} cvtdl_keypoint_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvtdl_keypoint_info_t *info;
} cvtdl_keypoint_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
} cvtdl_seg_t;

typedef struct {
  float x[2];
  float y[2];
  float score;
} cvtdl_lane_point_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  cvtdl_lane_point_t *lane;
  int lane_state;
} cvtdl_lane_t;

typedef void *cvtdl_handle_t;
typedef void *cvtdl_image_t;

#ifdef __cplusplus
}
#endif
#endif
