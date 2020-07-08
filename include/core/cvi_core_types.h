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
} cvi_detect_rect_t;

typedef struct {
  int8_t* ptr;
  uint32_t size;
  feature_type_e type;
} cvi_feature_t;

typedef struct {
  float* x;
  float* y;
  uint32_t size;
} cvi_pts_t;

#endif