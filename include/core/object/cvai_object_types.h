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
  uint64_t unique_id;
  cvai_bbox_t bbox;
  cvai_feature_t feature;
  int classes;
} cvai_object_info_t;

typedef struct {
  uint32_t size;
  uint32_t width;
  uint32_t height;
  cvai_object_info_t *info;
} cvai_object_t;

#endif