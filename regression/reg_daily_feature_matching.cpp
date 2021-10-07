#include "cviai.h"

#include <cvimath/cvimath_internal.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <limits>
#include <random>
#include "core/core/cvai_core_types.h"
#include "cviai_test.hpp"

namespace cviai {
namespace unitest {

// type traits for feature type. Currently, only int8 type are supported
//////////////////////////////////
template <feature_type_e>
struct FeatureTypeTrait;

template <>
struct FeatureTypeTrait<TYPE_INT8> {
  typedef int8_t type;
  typedef std::uniform_int_distribution<int8_t> generator;
};

template <>
struct FeatureTypeTrait<TYPE_FLOAT> {
  typedef float type;
  typedef std::uniform_real_distribution<float> generator;
};
///////////////////////////////////

template <feature_type_e FeatureTypeEnum>
struct GoldenResult {
  using DataType = typename FeatureTypeTrait<FeatureTypeEnum>::type;
  using Generator = typename FeatureTypeTrait<FeatureTypeEnum>::generator;

  GoldenResult(uint32_t num_db_features, uint32_t feature_length, uint32_t _topk) : topk(_topk) {
    db_feature.data_num = num_db_features;
    db_feature.feature_length = feature_length;
    db_feature.ptr = (int8_t *)malloc(db_feature.data_num * db_feature.feature_length);
    db_feature.type = FeatureTypeEnum;

    input_feature.ptr = (int8_t *)malloc(db_feature.feature_length);
    input_feature.size = db_feature.feature_length;
    input_feature.type = FeatureTypeEnum;
    topk_indices = (uint32_t *)malloc(topk * sizeof(uint32_t));
    topk_similarity = (float *)malloc(topk * sizeof(float));
    ;
  }

  ~GoldenResult() {
    free(db_feature.ptr);
    free(input_feature.ptr);
    free(topk_indices);
    free(topk_similarity);
  }

  void init() {
    typedef std::numeric_limits<DataType> data_limit;

    std::random_device rd;
    std::default_random_engine eng(rd());
    Generator distr(data_limit::min(), data_limit::max());

    // generate random features
    ///////////////////////////////////////////////
    for (uint32_t i = 0; i < input_feature.size; i++) {
      ((DataType *)input_feature.ptr)[i] = distr(eng);
    }

    for (uint32_t j = 0; j < db_feature.data_num; j++) {
      for (uint32_t i = 0; i < db_feature.feature_length; i++) {
        ((DataType *)db_feature.ptr)[j * db_feature.feature_length + i] = distr(eng);
      }
    }

    // generate golden matching result
    ///////////////////////////////////////////////
    float *db_unit = (float *)malloc(db_feature.data_num * sizeof(float));
    float *buffer_f = (float *)malloc(db_feature.data_num * sizeof(float));

    cvm_gen_precached_i8_unit_length((DataType *)db_feature.ptr, db_unit, db_feature.feature_length,
                                     db_feature.data_num);

    cvm_cpu_i8data_ip_match((DataType *)input_feature.ptr, (DataType *)db_feature.ptr, db_unit,
                            topk_indices, topk_similarity, buffer_f, db_feature.feature_length,
                            db_feature.data_num, topk);

    free(db_unit);
    free(buffer_f);
  }

  cvai_service_feature_array_t db_feature;
  cvai_feature_t input_feature;
  uint32_t topk;
  uint32_t *topk_indices;
  float *topk_similarity;
};

struct Similarity {
  float value;
  uint32_t index;
};

class FeatureMatchingTestSuite : public CVIAITestSuite {
 public:
  FeatureMatchingTestSuite() {}
  virtual ~FeatureMatchingTestSuite() = default;

 protected:
  virtual void SetUp() {
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 0, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_Service_CreateHandle(&m_service_handle, m_ai_handle), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_Service_DestroyHandle(m_service_handle);
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
    m_service_handle = NULL;
  }

  static int cmp(const void *a, const void *b);

  cviai_handle_t m_ai_handle;
  cviai_service_handle_t m_service_handle;
};

int FeatureMatchingTestSuite::cmp(const void *a, const void *b) {
  Similarity *a1 = (Similarity *)a;
  Similarity *a2 = (Similarity *)b;
  if ((*a1).value >= (*a2).value)
    return -1;
  else if ((*a1).value < (*a2).value)
    return 1;
  else
    return 0;
}

TEST_F(FeatureMatchingTestSuite, object_info_matching) {
  // test without register
  {
    cvai_object_info_t obj_info;
    float sims[10];
    uint32_t indices[10];
    uint32_t score_size;
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, 5, 0, indices, sims,
                                                &score_size),
              CVIAI_ERR_NOT_YET_INITIALIZED);
  }

  std::vector<uint32_t> num_features = {100, 500, 10000, 20000};
  for (uint32_t num_feat : num_features) {
    GoldenResult<TYPE_INT8> golden(num_feat, 512, 5);
    golden.init();

    std::vector<float> vec_sims(golden.topk);
    std::vector<uint32_t> vec_indices(golden.topk);

    float *sims = vec_sims.data();
    uint32_t *indices = vec_indices.data();

    uint32_t score_size;

    cvai_object_info_t obj_info;
    obj_info.feature = golden.input_feature;

    ASSERT_EQ(
        CVI_AI_Service_RegisterFeatureArray(m_service_handle, golden.db_feature, COS_SIMILARITY),
        CVIAI_SUCCESS);

    // test matching top-k similarity without threshold
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, golden.topk, 0,
                                                indices, sims, &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);

    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with threshold
    float threshold = golden.topk_similarity[golden.topk - 1];
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, 0, threshold, indices,
                                                sims, &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);

    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with top-k and threshold
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, golden.topk, threshold,
                                                indices, sims, &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);
    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with wrong top-k and threshold value
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, 0, 0, indices, sims,
                                                &score_size),
              CVIAI_ERR_INVALID_ARGS);

    // test matching with wrong type value
    obj_info.feature.type = TYPE_BF16;
    ASSERT_EQ(CVI_AI_Service_ObjectInfoMatching(m_service_handle, &obj_info, 0, 0, indices, sims,
                                                &score_size),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(FeatureMatchingTestSuite, face_info_matching) {
  // test without register
  {
    cvai_face_info_t face_info;
    float sims[10];
    uint32_t indices[10];
    uint32_t score_size;
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, 5, 0, indices, sims,
                                              &score_size),
              CVIAI_ERR_NOT_YET_INITIALIZED);
  }

  std::vector<uint32_t> num_features = {100, 500, 10000, 20000};
  for (uint32_t num_feat : num_features) {
    GoldenResult<TYPE_INT8> golden(num_feat, 512, 5);
    golden.init();

    ASSERT_EQ(
        CVI_AI_Service_RegisterFeatureArray(m_service_handle, golden.db_feature, COS_SIMILARITY),
        CVIAI_SUCCESS);

    std::vector<float> sims(golden.topk);
    std::vector<uint32_t> indices(golden.topk);

    uint32_t score_size;

    cvai_face_info_t face_info;
    face_info.feature = golden.input_feature;

    // test matching top-k similarity without threshold
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, golden.topk, 0,
                                              indices.data(), sims.data(), &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);

    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with threshold
    float threshold = golden.topk_similarity[golden.topk - 1];
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, 0, threshold,
                                              indices.data(), sims.data(), &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);

    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with top-k and threshold
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, golden.topk, threshold,
                                              indices.data(), sims.data(), &score_size),
              CVIAI_SUCCESS);

    EXPECT_EQ(score_size, golden.topk);
    for (uint32_t i = 0; i < golden.topk; i++) {
      EXPECT_EQ(indices[i], golden.topk_indices[i]);
      EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
    }

    // test matching with wrong top-k and threshold value
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, 0, 0, indices.data(),
                                              sims.data(), &score_size),
              CVIAI_ERR_INVALID_ARGS);

    // test matching with wrong type value
    face_info.feature.type = TYPE_BF16;
    ASSERT_EQ(CVI_AI_Service_FaceInfoMatching(m_service_handle, &face_info, 0, 0, indices.data(),
                                              sims.data(), &score_size),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(FeatureMatchingTestSuite, raw_matching) {
  GoldenResult<TYPE_INT8> golden(20000, 512, 5);
  golden.init();

  ASSERT_EQ(
      CVI_AI_Service_RegisterFeatureArray(m_service_handle, golden.db_feature, COS_SIMILARITY),
      CVIAI_SUCCESS);

  std::vector<float> vec_sims(golden.topk);
  std::vector<uint32_t> vec_indices(golden.topk);

  float *sims = vec_sims.data();
  uint32_t *indices = vec_indices.data();

  uint32_t score_size;
  ASSERT_EQ(CVI_AI_Service_RawMatching(m_service_handle, (uint8_t *)golden.input_feature.ptr,
                                       TYPE_INT8, golden.topk, 0, indices, sims, &score_size),
            CVIAI_SUCCESS);

  EXPECT_EQ(score_size, golden.topk);

  for (uint32_t i = 0; i < golden.topk; i++) {
    EXPECT_EQ(indices[i], golden.topk_indices[i]);
    EXPECT_FLOAT_EQ(sims[i], golden.topk_similarity[i]);
  }
}

TEST_F(FeatureMatchingTestSuite, calculate_similarity) {
  GoldenResult<TYPE_INT8> golden(20000, 512, 5);
  golden.init();

  std::vector<Similarity> sims(golden.db_feature.data_num);

  for (uint32_t i = 0; i < golden.db_feature.data_num; i++) {
    cvai_feature_t db_feature;
    db_feature.ptr = &((int8_t *)golden.db_feature.ptr)[i * golden.db_feature.feature_length];
    db_feature.size = golden.db_feature.feature_length;
    db_feature.type = TYPE_INT8;
    ASSERT_EQ(CVI_AI_Service_CalculateSimilarity(m_service_handle, &golden.input_feature,
                                                 &db_feature, &sims[i].value),
              CVIAI_SUCCESS);
    sims[i].index = i;
  }

  std::stable_sort(sims.begin(), sims.end(),
                   [](const Similarity &v1, const Similarity &v2) { return v1.value >= v2.value; });

  for (uint32_t i = 0; i < golden.topk; i++) {
    EXPECT_EQ(sims[i].index, golden.topk_indices[i]);
    EXPECT_FLOAT_EQ(sims[i].value, golden.topk_similarity[i]);
  }
}
}  // namespace unitest
}  // namespace cviai