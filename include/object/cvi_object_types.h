#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include "core/core_types.h"

typedef struct {
  char name[128];
  cvi_bbox_t bbox;
  int classes;
} cvi_object_info_t;

typedef struct {
  int size;
  int width;
  int height;
  cvi_object_info_t *objects;
} cvi_object_meta_t;

#endif