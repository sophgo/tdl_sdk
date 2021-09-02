#include "cviai.h"

#include <cvimath/cvimath_internal.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "core/core/cvai_core_types.h"

#define EXPECT_EQUAL_F(x, y)                                                          \
  do {                                                                                \
    if (fabs((x) - (y)) > 1e-6) {                                                     \
      printf("value not equal! expect: %.7f, actual: %.7f %s at (%s:%d)\n", (x), (y), \
             __FUNCTION__, __FILE__, __LINE__);                                       \
      return;                                                                         \
    }                                                                                 \
  } while (0)

#define EXPECT_EQUAL_U32(x, y)                                                                  \
  do {                                                                                          \
    if ((x) != (y)) {                                                                           \
      printf("value not equal! expect: %u, actual: %u %s at (%s:%d)\n", (x), (y), __FUNCTION__, \
             __FILE__, __LINE__);                                                               \
      return;                                                                                   \
    }                                                                                           \
  } while (0)

#define EXPECT_EQUAL_S32(x, y)                                                                  \
  do {                                                                                          \
    if ((x) != (y)) {                                                                           \
      printf("value not equal! expect: %d, actual: %d %s at (%s:%d)\n", (x), (y), __FUNCTION__, \
             __FILE__, __LINE__);                                                               \
      return;                                                                                   \
    }                                                                                           \
  } while (0)

#define PERF_BEGIN()     \
  struct timeval t0, t1; \
  gettimeofday(&t0, NULL);

#define PERF_END()                                                                       \
  gettimeofday(&t1, NULL);                                                               \
  unsigned long elapsed = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec); \
  printf("[PASSED] %s, elapsed: %lu us\n", __FUNCTION__, elapsed);

typedef struct _Similarity {
  float value;
  int index;
} Similarity;

int cmp(const void *a, const void *b) {
  Similarity *a1 = (Similarity *)a;
  Similarity *a2 = (Similarity *)b;
  if ((*a1).value >= (*a2).value)
    return -1;
  else if ((*a1).value < (*a2).value)
    return 1;
  else
    return 0;
}

void test_CVI_AI_Service_CalculateSimilarity(cviai_service_handle_t handle,
                                             cvai_feature_t *input_feature,
                                             cvai_service_feature_array_t *db_features,
                                             float *golden_sims, uint32_t *golden_indices,
                                             uint32_t golden_size) {
  PERF_BEGIN();

  Similarity *sims = (Similarity *)malloc(sizeof(Similarity) * db_features->data_num);

  for (int i = 0; i < db_features->data_num; i++) {
    cvai_feature_t db_feature;
    db_feature.ptr = &((int8_t *)db_features->ptr)[i * db_features->feature_length];
    db_feature.size = db_features->feature_length;
    db_feature.type = TYPE_INT8;
    int32_t ret =
        CVI_AI_Service_CalculateSimilarity(handle, input_feature, &db_feature, &sims[i].value);
    EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
    sims[i].index = i;
  }

  qsort(sims, db_features->data_num, sizeof(Similarity), cmp);

  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(sims[i].index, golden_indices[i]);
    EXPECT_EQUAL_F(sims[i].value, golden_sims[i]);
  }

  free(sims);

  PERF_END();
}

void test_CVI_AI_Service_RawMatching(cviai_service_handle_t handle, cvai_feature_t *input_feature,
                                     float *golden_sims, uint32_t *golden_indices,
                                     uint32_t golden_size) {
  PERF_BEGIN();
  float *sims = (float *)malloc(sizeof(float) * golden_size);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * golden_size);

  uint32_t score_size;
  int32_t ret = CVI_AI_Service_RawMatching(handle, (uint8_t *)input_feature->ptr, TYPE_INT8,
                                           golden_size, 0, indices, sims, &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);

  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  free(sims);
  free(indices);
  PERF_END();
}

void test_CVI_AI_Service_FaceInfoMatching(cviai_service_handle_t handle,
                                          cvai_feature_t *input_feature, float *golden_sims,
                                          uint32_t *golden_indices, uint32_t golden_size) {
  PERF_BEGIN();
  float *sims = (float *)malloc(sizeof(float) * golden_size);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * golden_size);

  uint32_t score_size;

  cvai_face_info_t face_info;
  face_info.feature = *input_feature;

  // test matching top-k similarity
  int32_t ret = CVI_AI_Service_FaceInfoMatching(handle, &face_info, golden_size, 0, indices, sims,
                                                &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);

  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  // test matching with threshold
  float threshold = golden_sims[golden_size - 1];
  ret =
      CVI_AI_Service_FaceInfoMatching(handle, &face_info, 0, threshold, indices, sims, &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);
  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  // test matching with top-k and threshold
  ret = CVI_AI_Service_FaceInfoMatching(handle, &face_info, golden_size, threshold, indices, sims,
                                        &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);
  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  free(sims);
  free(indices);

  PERF_END();
}

void test_CVI_AI_Service_ObjectInfoMatching(cviai_service_handle_t handle,
                                            cvai_feature_t *input_feature, float *golden_sims,
                                            uint32_t *golden_indices, uint32_t golden_size) {
  PERF_BEGIN();
  float *sims = (float *)malloc(sizeof(float) * golden_size);
  uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * golden_size);

  uint32_t score_size;

  cvai_object_info_t obj_info;
  obj_info.feature = *input_feature;

  // test matching top-k similarity
  int32_t ret = CVI_AI_Service_ObjectInfoMatching(handle, &obj_info, golden_size, 0, indices, sims,
                                                  &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);

  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  // test matching with threshold
  float threshold = golden_sims[golden_size - 1];
  ret = CVI_AI_Service_ObjectInfoMatching(handle, &obj_info, 0, threshold, indices, sims,
                                          &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);
  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  // test matching with top-k and threshold
  ret = CVI_AI_Service_ObjectInfoMatching(handle, &obj_info, golden_size, threshold, indices, sims,
                                          &score_size);
  EXPECT_EQUAL_S32(ret, CVIAI_SUCCESS);
  EXPECT_EQUAL_U32(score_size, golden_size);
  for (uint32_t i = 0; i < golden_size; i++) {
    EXPECT_EQUAL_U32(golden_indices[i], indices[i]);
    EXPECT_EQUAL_F(golden_sims[i], sims[i]);
  }

  free(sims);
  free(indices);

  PERF_END();
}

int main() {
  cviai_handle_t ai_handle;
  cviai_service_handle_t handle;
  CVI_AI_CreateHandle(&ai_handle);
  CVI_AI_Service_CreateHandle(&handle, ai_handle);

  cvai_service_feature_array_t featureArray;
  featureArray.data_num = 20000;
  featureArray.feature_length = 512;
  featureArray.ptr = (int8_t *)malloc(featureArray.data_num * featureArray.feature_length);
  featureArray.type = TYPE_INT8;

  cvai_feature_t input_feature;
  input_feature.ptr = (int8_t *)malloc(featureArray.feature_length);
  input_feature.size = featureArray.feature_length;
  input_feature.type = TYPE_INT8;

  // generate random features
  ///////////////////////////////////////////////
  srand(time(NULL));
  for (uint32_t i = 0; i < featureArray.feature_length; i++) {
    ((int8_t *)input_feature.ptr)[i] = rand() % 10 - 10;
  }
  for (uint32_t j = 0; j < featureArray.data_num; j++) {
    for (uint32_t i = 0; i < featureArray.feature_length; i++) {
      ((int8_t *)featureArray.ptr)[j * featureArray.feature_length + i] = rand() % 10 - 10;
    }
  }

  // generate golden matching result
  ///////////////////////////////////////////////
  struct timeval t0, t1;
  const uint32_t k = 5;
  uint32_t *index = (uint32_t *)malloc(k * sizeof(uint32_t));
  float *db_unit = (float *)malloc(featureArray.data_num * sizeof(float));
  float *buffer_f = (float *)malloc(featureArray.data_num * sizeof(float));
  gettimeofday(&t0, NULL);
  cvm_gen_precached_i8_unit_length((int8_t *)featureArray.ptr, db_unit, featureArray.feature_length,
                                   featureArray.data_num);
  float *value = (float *)malloc(k * sizeof(float));
  cvm_cpu_i8data_ip_match((int8_t *)input_feature.ptr, (int8_t *)featureArray.ptr, db_unit, index,
                          value, buffer_f, featureArray.feature_length, featureArray.data_num, k);
  gettimeofday(&t1, NULL);

  for (uint32_t i = 0; i < k; i++) {
    value[i] = value[i];
    printf("Golden matching result: [%u] %f\n", index[i], value[i]);
  }
  printf("------------------------------------\n");
  ///////////////////////////////////////////////

  CVI_AI_Service_RegisterFeatureArray(handle, featureArray, COS_SIMILARITY);

  test_CVI_AI_Service_RawMatching(handle, &input_feature, value, index, k);
  test_CVI_AI_Service_FaceInfoMatching(handle, &input_feature, value, index, k);
  test_CVI_AI_Service_ObjectInfoMatching(handle, &input_feature, value, index, k);
  test_CVI_AI_Service_CalculateSimilarity(handle, &input_feature, &featureArray, value, index, k);

  free(value);
  free(buffer_f);
  free(db_unit);

  free(index);
  free(input_feature.ptr);
  free(featureArray.ptr);
  CVI_AI_Service_DestroyHandle(handle);
  CVI_AI_DestroyHandle(ai_handle);
}