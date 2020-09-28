#ifndef _CVIAI_TYPES_MEM_INTERNAL_H_
#define _CVIAI_TYPES_MEM_INTERNAL_H_
#include "cviai_log.hpp"

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