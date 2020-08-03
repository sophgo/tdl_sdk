#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include "core/core/cvai_core_types.h"

typedef enum {
  CVI_DET_TYPE_ALL = 0,
  CVI_DET_TYPE_VEHICLE = (1 << 0),
  CVI_DET_TYPE_PEOPLE = (1 << 1),
  CVI_DET_TYPE_PET = (1 << 2)
} cvai_obj_det_type_t;

typedef struct {
  char name[128];
  cvai_bbox_t bbox;
  int classes;
} cvai_object_info_t;

typedef struct {
  int size;
  int width;
  int height;
  cvai_object_info_t *objects;
} cvai_object_t;

#endif