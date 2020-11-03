#ifndef _CVIAI_TYPES_MEM_INTERNAL_H_
#define _CVIAI_TYPES_MEM_INTERNAL_H_
#include "core/cviai_types_mem.h"
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"
#include "cviai_log.hpp"

#include <string.h>

inline void CVI_AI_MemAlloc(const uint32_t unit_len, const uint32_t size, const feature_type_e type,
                            cvai_feature_t *feature) {
  if (feature->size != size || feature->type != type) {
    free(feature->ptr);
    feature->ptr = (int8_t *)malloc(unit_len * size);
    feature->size = size;
    feature->type = type;
  }
}

inline void CVI_AI_MemAlloc(const uint32_t size, cvai_pts_t *pts) {
  if (pts->size != size) {
    free(pts->x);
    free(pts->y);
    pts->x = (float *)malloc(sizeof(float) * size);
    pts->y = (float *)malloc(sizeof(float) * size);
    pts->size = size;
  }
}

inline void CVI_AI_MemAlloc(const uint32_t size, cvai_tracker_t *tracker) {
  if (tracker->size != size) {
    free(tracker->info);
    tracker->info = (cvai_tracker_info_t *)malloc(size * sizeof(cvai_tracker_info_t));
    tracker->size = size;
  }
}

inline void CVI_AI_MemAlloc(const uint32_t size, cvai_face_t *meta) {
  if (meta->size != size) {
    for (uint32_t i = 0; i < meta->size; i++) {
      CVI_AI_FreeCpp(&meta->info[i]);
      free(meta->info);
    }
    meta->size = size;
    meta->info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * meta->size);
  }
}

inline void CVI_AI_MemAllocInit(const uint32_t size, const uint32_t pts_num, cvai_face_t *meta) {
  CVI_AI_MemAlloc(size, meta);
  for (uint32_t i = 0; i < meta->size; ++i) {
    meta->info[i].bbox.x1 = -1;
    meta->info[i].bbox.x2 = -1;
    meta->info[i].bbox.y1 = -1;
    meta->info[i].bbox.y2 = -1;

    meta->info[i].name[0] = '\0';
    meta->info[i].emotion = EMOTION_UNKNOWN;
    meta->info[i].gender = GENDER_UNKNOWN;
    meta->info[i].race = RACE_UNKNOWN;
    meta->info[i].age = -1;
    meta->info[i].liveness_score = -1;
    meta->info[i].mask_score = -1;
    memset(&meta->info[i].face_quality, 0, sizeof(cvai_face_quality_t));
    memset(&meta->info[i].feature, 0, sizeof(cvai_feature_t));
    if (pts_num > 0) {
      meta->info[i].pts.x = (float *)malloc(sizeof(float) * pts_num);
      meta->info[i].pts.y = (float *)malloc(sizeof(float) * pts_num);
      meta->info[i].pts.size = pts_num;
      for (uint32_t j = 0; j < meta->info[i].pts.size; ++j) {
        meta->info[i].pts.x[j] = -1;
        meta->info[i].pts.y[j] = -1;
      }
    } else {
      memset(&meta->info[i].pts, 0, sizeof(meta->info[i].pts));
    }
  }
}

inline void __attribute__((always_inline))
featurePtrConvert2Float(cvai_feature_t *feature, float *output) {
  switch (feature->type) {
    case TYPE_INT8: {
      int8_t *ptr = (int8_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_UINT8: {
      uint8_t *ptr = (uint8_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_INT16: {
      int16_t *ptr = (int16_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_UINT16: {
      uint16_t *ptr = (uint16_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_INT32: {
      int32_t *ptr = (int32_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_UINT32: {
      uint32_t *ptr = (uint32_t *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    case TYPE_BF16: {
      uint16_t *ptr = (uint16_t *)(feature->ptr);
      union {
        float a;
        uint16_t b[2];
      } bfb;
      bfb.b[0] = 0;
      for (uint32_t i = 0; i < feature->size; ++i) {
        bfb.b[1] = ptr[i];
        output[i] = bfb.a;
      }
    } break;
    case TYPE_FLOAT: {
      float *ptr = (float *)(feature->ptr);
      for (uint32_t i = 0; i < feature->size; ++i) {
        output[i] = ptr[i];
      }
    } break;
    default:
      LOGE("Unsupported type: %u.\n", feature->type);
      break;
  }
  return;
}

#endif  // End of _CVIAI_TYPES_MEM_INTERNAL_H_