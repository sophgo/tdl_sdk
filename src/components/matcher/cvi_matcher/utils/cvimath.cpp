#include "cvimath_internal.h"

#include <bits/stdc++.h>
#ifdef __ARM_ARCH
#include <arm_neon.h>
#endif

template <typename T>
void kSelectionSortIndex(T *array, uint32_t *index, T *value,
                         const uint32_t array_size, const uint32_t k) {
  for (uint32_t i = 0; i < k; i++) {
    int largest = 0;
    for (uint32_t j = 0; j < array_size; j++) {
      if (array[j] > array[largest]) {
        largest = j;
      }
    }
    value[i] = array[largest];
    index[i] = largest;
    array[largest] = 0;
  }
}

inline uint32_t dot(uint8_t *a, uint8_t *b, uint32_t data_length) {
  uint32_t dot_result = 0;
  for (uint32_t i = 0; i < data_length; i++) {
    dot_result += ((short)a[i] * b[i]);
  }
  return dot_result;
}

inline int32_t dotI8(int8_t *a, int8_t *b, uint32_t data_length) {
  int32_t dot_result = 0;
  for (uint32_t i = 0; i < data_length; i++) {
    dot_result += ((short)a[i] * b[i]);
  }
  return dot_result;
}

// 新增：计算浮点数组的点积
inline float dotF32(float *a, float *b, uint32_t data_length) {
  float dot_result = 0.0f;
  for (uint32_t i = 0; i < data_length; i++) {
    dot_result += (a[i] * b[i]);
  }
  return dot_result;
}

void cvmGenPrecachedI8UnitLength(int8_t *precached, float *unit_precached_arr,
                                 const uint32_t data_length,
                                 const uint32_t data_num) {
  for (uint32_t i = 0; i < data_num; i++) {
    int8_t *fb_offset = precached + i * data_length;
    unit_precached_arr[i] = dotI8(fb_offset, fb_offset, data_length);
    unit_precached_arr[i] = sqrt(unit_precached_arr[i]);
  }
}

void cvmGenPrecachedU8UnitLength(uint8_t *precached, float *unit_precached_arr,
                                 const uint32_t data_length,
                                 const uint32_t data_num) {
  for (uint32_t i = 0; i < data_num; i++) {
    uint8_t *fb_offset = precached + i * data_length;
    unit_precached_arr[i] = dot(fb_offset, fb_offset, data_length);
    unit_precached_arr[i] = sqrt(unit_precached_arr[i]);
  }
}

// 新增：计算FP32类型特征数组的单位长度
void cvmGenPrecachedFP32UnitLength(float *precached, float *unit_precached_arr,
                                   const uint32_t data_length,
                                   const uint32_t data_num) {
  for (uint32_t i = 0; i < data_num; i++) {
    float *fb_offset = precached + i * data_length;
    unit_precached_arr[i] = dotF32(fb_offset, fb_offset, data_length);
    unit_precached_arr[i] = sqrt(unit_precached_arr[i]);
  }
}

void cvmCpuI8dataIpMatch(int8_t *feature, int8_t *precached,
                         float *unit_precached_arr, uint32_t *k_index,
                         float *k_value, float *buffer,
                         const uint32_t data_length, const uint32_t data_num,
                         const uint32_t k) {
  float unit_feature = (float)dotI8(feature, feature, data_length);
  unit_feature = sqrt(unit_feature);
  for (uint32_t i = 0; i < data_num; i++) {
    buffer[i] = dotI8(feature, precached + i * data_length, data_length) /
                (unit_feature * unit_precached_arr[i]);
  }
  kSelectionSortIndex(buffer, k_index, k_value, data_num, k);
}

void cvmCpuU8dataIpMatch(uint8_t *feature, uint8_t *precached,
                         float *unit_precached_arr, uint32_t *k_index,
                         float *k_value, float *buffer,
                         const uint32_t data_length, const uint32_t data_num,
                         const uint32_t k) {
  float unit_feature = (float)dot(feature, feature, data_length);
  unit_feature = sqrt(unit_feature);
  for (uint32_t i = 0; i < data_num; i++) {
    buffer[i] = dot(feature, precached + i * data_length, data_length) /
                (unit_feature * unit_precached_arr[i]);
  }
  kSelectionSortIndex(buffer, k_index, k_value, data_num, k);
}

// 新增：FP32类型的余弦相似度计算
void cvmCpuFP32dataIpMatch(float *feature, float *precached,
                           float *unit_precached_arr, uint32_t *k_index,
                           float *k_value, float *buffer,
                           const uint32_t data_length, const uint32_t data_num,
                           const uint32_t k) {
  float unit_feature = dotF32(feature, feature, data_length);
  unit_feature = sqrt(unit_feature);
  for (uint32_t i = 0; i < data_num; i++) {
    buffer[i] = dotF32(feature, precached + i * data_length, data_length) /
                (unit_feature * unit_precached_arr[i]);
  }
  kSelectionSortIndex(buffer, k_index, k_value, data_num, k);
}