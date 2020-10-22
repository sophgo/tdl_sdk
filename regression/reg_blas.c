#include "cviai.h"

#include <cvimath/cvimath_internal.h>
#include <time.h>

int main() {
  cviai_handle_t ai_handle;
  cviai_service_handle_t handle;
  CVI_AI_CreateHandle(&ai_handle);
  CVI_AI_Service_CreateHandle(&handle, ai_handle);

  cvai_service_feature_array_t featureArray;
  featureArray.data_num = 1000;
  featureArray.feature_length = 512;
  featureArray.ptr = (int8_t *)malloc(featureArray.data_num * featureArray.feature_length);
  featureArray.type = TYPE_INT8;
  int8_t *input_feature = (int8_t *)malloc(featureArray.feature_length);

  srand(time(NULL));
  for (uint32_t i = 0; i < featureArray.feature_length; i++) {
    ((int8_t *)input_feature)[i] = rand() % 10 - 10;
  }
  for (uint32_t j = 0; j < featureArray.data_num; j++) {
    for (uint32_t i = 0; i < featureArray.feature_length; i++) {
      ((int8_t *)featureArray.ptr)[j * featureArray.feature_length + i] = rand() % 10 - 10;
    }
  }

  CVI_AI_Service_RegisterFeatureArray(handle, featureArray, INNER_PRODUCT);
  const uint32_t k = 5;
  uint32_t *index = (uint32_t *)malloc(k * sizeof(uint32_t));
  CVI_AI_Service_RawMatching(handle, (uint8_t *)input_feature, TYPE_INT8, k, &index);
  for (uint32_t i = 0; i < k; i++) {
    printf("[%u]\n", index[i]);
  }
  printf("\n");

  float *db_unit = (float *)malloc(featureArray.data_num * sizeof(float));
  float *buffer_f = (float *)malloc(featureArray.data_num * sizeof(float));
  cvm_gen_precached_i8_unit_length((int8_t *)featureArray.ptr, db_unit, featureArray.feature_length,
                                   featureArray.data_num);
  float *value = (float *)malloc(k * sizeof(float));
  cvm_cpu_i8data_ip_match((int8_t *)input_feature, (int8_t *)featureArray.ptr, db_unit, index,
                          value, buffer_f, featureArray.feature_length, featureArray.data_num, k);

  for (uint32_t i = 0; i < k; i++) {
    printf("[%u] %f\n", index[i], value[i]);
  }
  free(value);
  free(buffer_f);
  free(db_unit);

  free(index);
  free(input_feature);
  free(featureArray.ptr);
  CVI_AI_Service_DestroyHandle(handle);
  CVI_AI_DestroyHandle(ai_handle);
}