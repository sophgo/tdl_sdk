#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_
#include "core/core/cvai_core_types.h"

typedef enum {
  CVI_DET_TYPE_ALL = 0,
  CVI_DET_TYPE_VEHICLE = (1 << 0),
  CVI_DET_TYPE_PEOPLE = (1 << 1),
  CVI_DET_TYPE_PET = (1 << 2)
} cvai_obj_det_type_t;

typedef enum {
  CVI_TRACKER_NEW = 0,
  CVI_TRACKER_UNSTABLE,
  CVI_TRACKER_STABLE,
} cvai_trk_state_type_t;

typedef struct {
  cvai_trk_state_type_t state;
  // cvai_bbox_t bbox;    /* tracker computed bbox */
} cvai_tracker_info_t;

typedef struct {
  uint32_t size;
  cvai_tracker_info_t *info;
} cvai_tracker_t;

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