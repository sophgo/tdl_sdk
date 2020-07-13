#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include "core/cvi_core_types.h"

typedef enum {
  CVI_DET_TYPE_ALL = 0,
  CVI_DET_TYPE_VEHICLE = (1 << 0),
  CVI_DET_TYPE_PEOPLE = (1 << 1),
  CVI_DET_TYPE_PET = (1 << 2)
} cvi_obj_det_type_t;

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
} cvi_object_t;

#endif