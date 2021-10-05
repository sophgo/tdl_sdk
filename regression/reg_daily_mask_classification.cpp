#include <experimental/filesystem>
#include <fstream>
#include <memory>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "gtest.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cviai {
namespace unitest {

class MaskClassification : public CVIAIModelTestSuite {
 public:
  MaskClassification()
      : CVIAIModelTestSuite("daily_reg_MaskClassification.json", "reg_daily_mask_classification") {}

  virtual ~MaskClassification() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["reg_config"][0]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    m_ai_handle = NULL;
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }
};

TEST_F(MaskClassification, open_close_model) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  const char *model_path_get =
      CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));
}

TEST_F(MaskClassification, get_vpss_config) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());
  cvai_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;

  // CVI_AI_GetVpssChnConfig for CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION should be failed.
  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION, 100,
                                    100, 0, &vpssconfig),
            CVIAI_ERR_GET_VPSS_CHN_CONFIG);

  // make sure doesn't modify vpss config after CVI_AI_GetVpssChnConfig is called.
  EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);
}

TEST_F(MaskClassification, skip_vpss_preprocess) {
  std::string image_path =
      (m_image_dir / std::string(m_json_object["reg_config"][0]["test_images"][0])).string();
  Image image(image_path, PIXEL_FORMAT_RGB_888);
  ASSERT_TRUE(image.open());

  {
    // test inference with skip vpss = false
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);
    ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image.getFrame(), face_meta), CVIAI_SUCCESS);
    EXPECT_TRUE(face_meta->info[0].mask_score != -1.0);
  }

  {
    // test inference with skip vpss = true
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                           m_model_path.c_str(), true);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);

    // This operatation shoule be fail, because face quality needs vpss preprocessing.
    ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image.getFrame(), face_meta),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(MaskClassification, inference) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  std::string image_path =
      (m_image_dir / std::string(m_json_object["reg_config"][0]["test_images"][0])).string();

  // test inference for PIXEL_FORMAT_RGB_888 format.
  {
    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image_rgb.open());

    // test 1 face
    {
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 1);
      ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image_rgb.getFrame(), face_meta),
                CVIAI_SUCCESS);
      EXPECT_TRUE(face_meta->info[0].mask_score != -1.0);
    }

    // test 10 faces
    {
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 10);
      ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image_rgb.getFrame(), face_meta),
                CVIAI_SUCCESS);
      EXPECT_TRUE(face_meta->info[0].mask_score != -1.0);
      float score = face_meta->info[0].mask_score;
      for (uint32_t fid = 1; fid < 10; fid++) {
        EXPECT_FLOAT_EQ(score, face_meta->info[fid].mask_score);
      }
    }
  }

  // inference for PIXEL_FORMAT_RGB_888_PLANAR format.
  {
    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
    ASSERT_TRUE(image_rgb.open());
    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);
    ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image_rgb.getFrame(), face_meta),
              CVIAI_SUCCESS);
    EXPECT_TRUE(face_meta->info[0].mask_score != -1.0);
  }
}

TEST_F(MaskClassification, accruacy) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_MASKCLASSIFICATION,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  std::ofstream m_ofs_results;

  int img_num = int(m_json_object["reg_config"][0]["image_num"]);

  float threshold = float(m_json_object["reg_config"][0]["threshold"]);

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path =
        (m_image_dir / std::string(m_json_object["reg_config"][0]["test_images"][img_idx]))
            .string();
    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image_rgb.open());

    int expected_res = int(m_json_object["reg_config"][0]["expected_results"][img_idx]);

    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);

    ASSERT_EQ(CVI_AI_MaskClassification(m_ai_handle, image_rgb.getFrame(), face_meta),
              CVIAI_SUCCESS);

    if (expected_res == 0) {
      EXPECT_LT(face_meta->info[0].mask_score, threshold) << "image path: " << image_path << "\n"
                                                          << "model path: " << m_model_path;
    } else {
      EXPECT_GT(face_meta->info[0].mask_score, threshold) << "image path: " << image_path << "\n"
                                                          << "model path: " << m_model_path;
    }
  }
}

}  // namespace unitest
}  // namespace cviai