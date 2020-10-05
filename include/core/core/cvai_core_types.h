#ifndef _CVI_CORE_TYPES_H_
#define _CVI_CORE_TYPES_H_

#include <stdint.h>
#include <stdlib.h>

typedef enum {
  TYPE_INT8 = 0,
  TYPE_UINT8,
  TYPE_INT16,
  TYPE_UINT16,
  TYPE_INT32,
  TYPE_UINT32,
  TYPE_BF16,
  TYPE_FLOAT
} feature_type_e;

typedef struct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
} cvai_bbox_t;

typedef struct {
  int8_t* ptr;
  uint32_t size;
  feature_type_e type;
} cvai_feature_t;

typedef struct {
  float* x;
  float* y;
  uint32_t size;
} cvai_pts_t;

typedef enum {
  CVI_TRACKER_NEW = 0,
  CVI_TRACKER_UNSTABLE,
  CVI_TRACKER_STABLE,
} cvai_trk_state_type_t;

typedef struct {
  cvai_trk_state_type_t state;
  // cvai_bbox_t bbox;    /* Reserved tracker computed bbox */
} cvai_tracker_info_t;

typedef struct {
  uint32_t size;
  cvai_tracker_info_t* info;
} cvai_tracker_t;

inline const int getFeatureTypeSize(feature_type_e type) {
  uint32_t size = 1;
  switch (type) {
    case TYPE_INT8:
    case TYPE_UINT8:
      break;
    case TYPE_INT16:
    case TYPE_UINT16:
    case TYPE_BF16:
      size = 2;
      break;
    case TYPE_INT32:
    case TYPE_UINT32:
    case TYPE_FLOAT:
      size = 4;
      break;
  }
  return size;
}

#endif