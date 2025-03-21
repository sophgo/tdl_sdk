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
} tdl_data_type_e;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
} tdl_box_t;

typedef struct {
  int8_t *ptr;
  uint32_t size;
  tdl_data_type_e type;
} tdl_feature_t;

typedef struct {
  float *x;
  float *y;
  uint32_t size;
  float score;
} tdl_pts_t;

typedef struct {
  float x;
  float y;
  float score;
} tdl_landmark_info_t;

typedef struct {
  tdl_box_t box;
  float score;
  int class_id;
  tdl_landmark_info_t *landmark_properity;
  tdl_object_type_e obj_type;
} tdl_object_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  tdl_object_info_t *info;
} tdl_object_t;

typedef struct {
  char name[128];
  float score;
  uint64_t track_id;
  tdl_box_t box;
  tdl_pts_t landmarks;
  tdl_feature_t feature;

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
} tdl_face_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  tdl_face_info_t *info;
} tdl_face_t;

typedef struct {
  int32_t class_id;
  float score;
} tdl_class_info_t;

typedef struct {
  uint32_t size;

  tdl_class_info_t *info;
} tdl_class_t;

typedef struct {
  float x;
  float y;
  float score;
} tdl_keypoint_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  tdl_keypoint_info_t *info;
} tdl_keypoint_t;

typedef struct {
  uint32_t width;
  uint32_t height;
  uint32_t output_width;
  uint32_t output_height;
  uint8_t *class_id;
  uint8_t *class_conf;
} tdl_seg_t;

typedef struct {
  uint8_t *mask;
  float *mask_point;
  uint32_t mask_point_size;
  tdl_object_info_t *obj_info;
} tdl_instance_seg_info_t;


typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  uint32_t mask_width;
  uint32_t mask_height;
  tdl_instance_seg_info_t *info;
} tdl_instance_seg_t;

typedef struct {
  float x[2];
  float y[2];
  float score;
} tdl_lane_point_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;

  tdl_lane_point_t *lane;
  int lane_state;
} tdl_lane_t;

typedef struct {
  int w;
  int h;
  int8_t *int_logits;
} tdl_depth_logits_t;

typedef struct {
  uint32_t size;
  uint64_t id;
  tdl_box_t bbox;
  int out_num;
} tdl_tracker_t;

typedef void *tdl_handle_t;
typedef void *tdl_image_t;

#ifdef __cplusplus
}
#endif
#endif
