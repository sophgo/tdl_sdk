#pragma once

#include "cviai_log.hpp"
#include "service/cviai_service_types.h"

#include <cvimath/cvimath.h>

typedef struct {
  cvai_service_feature_array_t feature_array;
  float *feature_unit_length = nullptr;
  float *feature_array_buffer = nullptr;
} cvai_service_feature_array_ext_t;

inline void __attribute__((always_inline))
FreeFeatureArrayExt(cvai_service_feature_array_ext_t *feature_array_ext) {
  if (feature_array_ext->feature_unit_length != nullptr) {
    delete feature_array_ext->feature_unit_length;
    feature_array_ext->feature_unit_length = nullptr;
  }
  if (feature_array_ext->feature_array_buffer != nullptr) {
    delete feature_array_ext->feature_array_buffer;
    feature_array_ext->feature_array_buffer = nullptr;
  }
  if (feature_array_ext->feature_array.ptr != NULL) {
    free(feature_array_ext->feature_array.ptr);
    feature_array_ext->feature_array.ptr = NULL;
  }
}

inline int __attribute__((always_inline))
RegisterIPFeatureArray(const cvai_service_feature_array_t feature_array,
                       cvai_service_feature_array_ext_t *feature_array_ext) {
  float *unit_length = new float[feature_array.feature_length * feature_array.data_num];
  switch (feature_array.type) {
    case TYPE_INT8: {
      cvm_gen_precached_i8_unit_length((int8_t *)feature_array.ptr, unit_length,
                                       feature_array.feature_length, feature_array.data_num);
    } break;
    case TYPE_UINT8: {
      cvm_gen_precached_u8_unit_length((uint8_t *)feature_array.ptr, unit_length,
                                       feature_array.feature_length, feature_array.data_num);
    } break;
    default: {
      LOGE("Unsupported register data type %x.\n", feature_array.type);
      delete[] unit_length;
      return CVI_FAILURE;
    } break;
  }
  FreeFeatureArrayExt(feature_array_ext);
  feature_array_ext->feature_array = feature_array;
  feature_array_ext->feature_unit_length = unit_length;
  feature_array_ext->feature_array_buffer =
      new float[feature_array.feature_length * feature_array.data_num];
  return CVI_SUCCESS;
}

inline int __attribute__((always_inline))
FeatureMatchingIPRaw(const uint8_t *feature, const feature_type_e &type, const uint32_t k,
                     uint32_t **index, cvai_service_feature_array_ext_t *feature_array_ext) {
  if (feature_array_ext->feature_array_buffer == nullptr) {
    LOGE("Feature array not registered yet.\n");
    return CVI_FAILURE;
  }
  if (feature_array_ext->feature_array.type != type) {
    LOGE("The registered feature array type %x is not the same as the input type %x.\n",
         feature_array_ext->feature_array.type, type);
    return CVI_FAILURE;
  }
  uint32_t *k_index = (uint32_t *)malloc(sizeof(uint32_t) * k);
  float *k_value = (float *)malloc(sizeof(float) * k);
  switch (type) {
    case TYPE_INT8: {
      cvm_cpu_i8data_ip_match((int8_t *)feature, (int8_t *)feature_array_ext->feature_array.ptr,
                              feature_array_ext->feature_unit_length, k_index, k_value,
                              feature_array_ext->feature_array_buffer,
                              feature_array_ext->feature_array.feature_length,
                              feature_array_ext->feature_array.data_num, k);
    } break;
    case TYPE_UINT8: {
      cvm_cpu_u8data_ip_match((uint8_t *)feature, (uint8_t *)feature_array_ext->feature_array.ptr,
                              feature_array_ext->feature_unit_length, k_index, k_value,
                              feature_array_ext->feature_array_buffer,
                              feature_array_ext->feature_array.feature_length,
                              feature_array_ext->feature_array.data_num, k);
    } break;
    default: {
      LOGE("Unsupported register data type %x.\n", type);
      free(k_index);
      free(k_value);
      return CVI_FAILURE;
    } break;
  }
  *index = k_index;
  free(k_value);

  return CVI_SUCCESS;
}
