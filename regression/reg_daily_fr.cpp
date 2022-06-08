#include <fstream>
#include <string>
#include <unordered_map>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace cviai {
namespace unitest {

typedef enum {
  FaceRecognition,
  FaceAttribute,
  MaskFR,
} ModelType;

class FaceRecognitionTestSuite : public CVIAIModelTestSuite {
 public:
  typedef CVI_S32 (*InferenceFunc)(const cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_face_t *);
  struct ModelInfo {
    InferenceFunc inference;
    CVI_AI_SUPPORTED_MODEL_E index;
    std::string model_path;
  };

  FaceRecognitionTestSuite() : CVIAIModelTestSuite("daily_reg_FR.json", "reg_daily_fr") {}

  virtual ~FaceRecognitionTestSuite() = default;

 protected:
  virtual void SetUp() {
    m_ai_handle = NULL;
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_Service_CreateHandle(&m_service_handle, m_ai_handle), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_Service_DestroyHandle(m_service_handle);
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
    m_service_handle = NULL;
  }

  cviai_service_handle_t m_service_handle;
  ModelInfo getModel(ModelType model_type, const std::string &model_name);
};

FaceRecognitionTestSuite::ModelInfo FaceRecognitionTestSuite::getModel(
    ModelType model_type, const std::string &model_name) {
  ModelInfo model_info;
  model_info.index = CVI_AI_SUPPORTED_MODEL_END;

  std::string model_path = (m_model_dir / model_name).string();

  switch (model_type) {
    case FaceRecognition: {
      model_info.index = CVI_AI_SUPPORTED_MODEL_FACERECOGNITION;
      model_info.inference = CVI_AI_FaceRecognition;
    } break;
    case FaceAttribute: {
      model_info.index = CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE;
      model_info.inference = CVI_AI_FaceAttribute;
    } break;
    case MaskFR: {
      model_info.index = CVI_AI_SUPPORTED_MODEL_MASKFACERECOGNITION;
      model_info.inference = CVI_AI_MaskFaceRecognition;
    } break;
    default:
      printf("unsupported model type: %d\n", model_type);
  }
  model_info.model_path = model_path;
  return model_info;
}

TEST_F(FaceRecognitionTestSuite, open_close_model) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());
    ModelType model_type = test_config["model_type"];

    ModelInfo model_info = getModel(model_type, model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, model_info.index);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, model_info.model_path,
                 std::string(model_path_get));
  }
}

TEST_F(FaceRecognitionTestSuite, get_vpss_config) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());
    ModelType model_type = test_config["model_type"];

    ModelInfo model_info = getModel(model_type, model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    cvai_vpssconfig_t vpssconfig;
    vpssconfig.chn_attr.u32Height = 200;
    vpssconfig.chn_attr.u32Width = 200;
    vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
    vpssconfig.chn_attr.stNormalize.bEnable = false;

    // CVI_AI_GetVpssChnConfig for face recognition models should be failed.
    EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, model_info.index, 100, 100, 0, &vpssconfig),
              CVIAI_ERR_GET_VPSS_CHN_CONFIG);

    // make sure doesn't modify vpss config after CVI_AI_GetVpssChnConfig is called.
    EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
    EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
    EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
    EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);
  }
}

TEST_F(FaceRecognitionTestSuite, skip_vpss_preprocess) {
  std::string image_path =
      (m_image_dir / std::string(m_json_object[0]["same_pairs"][0][0])).string();
  Image image(image_path, PIXEL_FORMAT_RGB_888);
  ASSERT_TRUE(image.open());

  std::string model_name = std::string(std::string(m_json_object[0]["model_name"]).c_str());
  ModelType model_type = m_json_object[0]["model_type"];

  ModelInfo model_info = getModel(model_type, model_name);

  {
    // test inference with skip vpss = false
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);
    ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), face_meta), CVIAI_SUCCESS);
    EXPECT_TRUE(face_meta->info[0].feature.ptr != NULL);
    EXPECT_GT(face_meta->info[0].feature.size, (uint32_t)0);
  }

  {
    // test inference with skip vpss = true
    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), true);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);

    // This operatation shoule be fail, because face recognition model needs vpss preprocessing.
    ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), face_meta),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(FaceRecognitionTestSuite, inference) {
  std::string model_name = std::string(std::string(m_json_object[0]["model_name"]).c_str());
  ModelType model_type = m_json_object[0]["model_type"];

  ModelInfo model_info = getModel(model_type, model_name);
  AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  std::string image_path =
      (m_image_dir / std::string(m_json_object[0]["same_pairs"][0][0])).string();

  // test inference for PIXEL_FORMAT_RGB_888 format.
  {
    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image_rgb.open());

    // test 1 face
    {
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 1);
      ASSERT_EQ(model_info.inference(m_ai_handle, image_rgb.getFrame(), face_meta), CVIAI_SUCCESS);
      EXPECT_TRUE(face_meta->info[0].feature.ptr != NULL);
      EXPECT_GT(face_meta->info[0].feature.size, (uint32_t)0);
    }

    // test 10 faces
    {
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 10);
      ASSERT_EQ(model_info.inference(m_ai_handle, image_rgb.getFrame(), face_meta), CVIAI_SUCCESS);
      EXPECT_TRUE(face_meta->info[0].feature.ptr != NULL);
      EXPECT_GT(face_meta->info[0].feature.size, (uint32_t)0);
      for (uint32_t fid = 1; fid < 10; fid++) {
        for (uint32_t feat_id = 0; feat_id < face_meta->info[0].feature.size; feat_id++) {
          EXPECT_EQ(face_meta->info[fid].feature.ptr[feat_id],
                    face_meta->info[0].feature.ptr[feat_id]);
        }
      }
    }
  }

  // inference for PIXEL_FORMAT_RGB_888_PLANAR format should be failed.
  {
    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
    ASSERT_TRUE(image_rgb.open());
    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);

    ASSERT_EQ(model_info.inference(m_ai_handle, image_rgb.getFrame(), face_meta),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(FaceRecognitionTestSuite, accuracy) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name = std::string(std::string(test_config["model_name"]).c_str());
    ModelType model_type = test_config["model_type"];

    ModelInfo model_info = getModel(model_type, model_name);
    ASSERT_LT(model_info.index, CVI_AI_SUPPORTED_MODEL_END);

    AIModelHandler aimodel(m_ai_handle, model_info.index, model_info.model_path, false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    std::vector<std::pair<std::string, std::string>> pair_info = {
        {"same_pairs", "same_scores"},
        {"diff_pairs", "diff_scores"},
    };

    for (auto pair_test : pair_info) {
      for (size_t pair_idx = 0; pair_idx < test_config[pair_test.first].size(); pair_idx++) {
        auto pair = test_config[pair_test.first][pair_idx];
        float expected_score = test_config[pair_test.second][pair_idx];

        std::string image_path1 = (m_image_dir / std::string(pair[0])).string();
        std::string image_path2 = (m_image_dir / std::string(pair[1])).string();

        AIObject<cvai_face_t> face_meta1;
        {
          Image image(image_path1, PIXEL_FORMAT_RGB_888);
          ASSERT_NO_FATAL_FAILURE(image.open());
          init_face_meta(face_meta1, 1);
          ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), face_meta1), CVIAI_SUCCESS);
        }

        AIObject<cvai_face_t> face_meta2;
        {
          Image image(image_path2, PIXEL_FORMAT_RGB_888);
          ASSERT_NO_FATAL_FAILURE(image.open());
          init_face_meta(face_meta2, 1);
          ASSERT_EQ(model_info.inference(m_ai_handle, image.getFrame(), face_meta2), CVIAI_SUCCESS);
        }

        float score = 0.0;
        CVI_AI_Service_CalculateSimilarity(m_service_handle, &face_meta1->info[0].feature,
                                           &face_meta2->info[0].feature, &score);

        EXPECT_LT(std::abs(score - expected_score), 0.1);
        CVI_AI_FreeCpp(face_meta1);
        CVI_AI_FreeCpp(face_meta2);
      }
    }
  }
}
}  // namespace unitest
}  // namespace cviai
