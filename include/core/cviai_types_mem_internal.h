#ifndef _CVIAI_TYPES_MEM_INTERNAL_H_
#define _CVIAI_TYPES_MEM_INTERNAL_H_
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

inline void CVI_AI_FaceInfoCopyToNew(const cvai_face_info_t *info, cvai_face_info_t *infoNew) {
  memcpy(infoNew->name, info->name, sizeof(info->name));
  infoNew->unique_id = info->unique_id;
  infoNew->bbox = info->bbox;
  infoNew->face_pts.size = info->face_pts.size;
  uint32_t pts_size = infoNew->face_pts.size * sizeof(float);
  infoNew->face_pts.x = (float *)malloc(pts_size);
  infoNew->face_pts.y = (float *)malloc(pts_size);
  memcpy(infoNew->face_pts.x, info->face_pts.x, pts_size);
  memcpy(infoNew->face_pts.y, info->face_pts.y, pts_size);
  infoNew->face_feature.size = info->face_feature.size;
  infoNew->face_feature.type = info->face_feature.type;
  uint32_t feature_size =
      infoNew->face_feature.size * getFeatureTypeSize(infoNew->face_feature.type);
  infoNew->face_feature.ptr = (int8_t *)malloc(feature_size);
  memcpy(infoNew->face_feature.ptr, info->face_feature.ptr, feature_size);
  infoNew->emotion = info->emotion;
  infoNew->gender = info->gender;
  infoNew->race = info->race;
  infoNew->age = info->age;
  infoNew->liveness_score = info->liveness_score;
  infoNew->mask_score = info->mask_score;
  infoNew->face_quality = info->face_quality;
}

inline void CVI_AI_ObjInfoCopyToNew(const cvai_object_info_t *info, cvai_object_info_t *infoNew) {
  memcpy(infoNew->name, info->name, sizeof(info->name));
  infoNew->unique_id = info->unique_id;
  infoNew->bbox = info->bbox;
  infoNew->feature.size = info->feature.size;
  infoNew->feature.type = info->feature.type;
  uint32_t feature_size = infoNew->feature.size * getFeatureTypeSize(infoNew->feature.type);
  infoNew->feature.ptr = (int8_t *)malloc(feature_size);
  memcpy(infoNew->feature.ptr, info->feature.ptr, feature_size);
  infoNew->classes = info->classes;
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