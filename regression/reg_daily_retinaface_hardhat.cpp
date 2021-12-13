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

class RetinafaceHardhatTestSuite : public CVIAIModelTestSuite {
 public:
  RetinafaceHardhatTestSuite()
      : CVIAIModelTestSuite("reg_daily_retinaface_hardhat.json", "reg_daily_retinaface_hardhat") {}

  virtual ~RetinafaceHardhatTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model"]);
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

TEST_F(RetinafaceHardhatTestSuite, open_close_model) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  const char *model_path_get =
      CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));
}

TEST_F(RetinafaceHardhatTestSuite, get_vpss_config) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());
  cvai_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;

  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT, 342,
                                    608, 0, &vpssconfig),
            CVIAI_SUCCESS);

  uint32_t dstWidth = 1280;
  uint32_t dstHeight = 720;
  float factor[3] = {0.84685433, 0.84685433, 0.84685433};
  float mean[3] = {104.16309, 99.081955, 88.072853};

  EXPECT_EQ(vpssconfig.chn_attr.u32Width, dstWidth);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, dstWidth);
  EXPECT_EQ(vpssconfig.chn_attr.u32Height, dstHeight);
  EXPECT_EQ(vpssconfig.chn_attr.enVideoFormat, VIDEO_FORMAT_LINEAR);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_BGR_888_PLANAR);

  for (uint32_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.factor[i], factor[i]);
  }
  for (uint32_t i = 0; i < 3; i++) {
    EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.mean[i], mean[i]);
  }
}

TEST_F(RetinafaceHardhatTestSuite, skip_vpss_preprocess) {
  // select image_0 for test
  std::string image_path = (m_image_dir / std::string(m_json_object["test_images"][0])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  {
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));
    EXPECT_EQ(CVI_AI_RetinaFace_Hardhat(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS);
  }
  {
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                           m_model_path.c_str(), true);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    AIObject<cvai_face_t> face_meta;
    init_face_meta(face_meta, 1);
    EXPECT_EQ(CVI_AI_RetinaFace_Hardhat(m_ai_handle, frame.getFrame(), face_meta),
              CVIAI_ERR_INFERENCE);
  }
}

TEST_F(RetinafaceHardhatTestSuite, inference) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  for (int img_idx = 0; img_idx < 1; img_idx++) {
    // select image_0 for test
    std::string image_path =
        (m_image_dir / std::string(m_json_object["test_images"][img_idx])).string();

    {
      Image frame(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
      ASSERT_TRUE(frame.open());

      cvai_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvai_face_t));
      EXPECT_EQ(CVI_AI_RetinaFace_Hardhat(m_ai_handle, frame.getFrame(), &face_meta),
                CVIAI_SUCCESS);
    }

    {
      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());

      cvai_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvai_face_t));
      EXPECT_EQ(CVI_AI_RetinaFace_Hardhat(m_ai_handle, frame.getFrame(), &face_meta),
                CVIAI_SUCCESS);
    }
  }
}

TEST_F(RetinafaceHardhatTestSuite, accruacy) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_HARDHAT,
                         m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  int img_num = int(m_json_object["test_images"].size());
  float threshold_bbox = float(m_json_object["threshold_bbox"]);
  float threshold_score = float(m_json_object["threshold_score"]);

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    // select image_0 for test
    std::string image_path =
        (m_image_dir / std::string(m_json_object["test_images"][img_idx])).string();

    Image frame(image_path, PIXEL_FORMAT_BGR_888);
    ASSERT_TRUE(frame.open());

    cvai_face_t face_meta;
    memset(&face_meta, 0, sizeof(cvai_face_t));

    {
      EXPECT_EQ(CVI_AI_RetinaFace_Hardhat(m_ai_handle, frame.getFrame(), &face_meta),
                CVIAI_SUCCESS);
    }

    for (uint32_t i = 0; i < face_meta.size; i++) {
      float expected_res_x1 = float(m_json_object["expected_results"][img_idx][1][i][0]);
      float expected_res_y1 = float(m_json_object["expected_results"][img_idx][1][i][1]);
      float expected_res_x2 = float(m_json_object["expected_results"][img_idx][1][i][2]);
      float expected_res_y2 = float(m_json_object["expected_results"][img_idx][1][i][3]);
      float expected_res_bbox_conf = float(m_json_object["expected_results"][img_idx][1][i][4]);
      float expected_res_hardhat_score = float(m_json_object["expected_results"][img_idx][1][i][5]);

      {
        EXPECT_LT(abs(face_meta.info[i].bbox.x1 - expected_res_x1), threshold_bbox);
        EXPECT_LT(abs(face_meta.info[i].bbox.y1 - expected_res_y1), threshold_bbox);
        EXPECT_LT(abs(face_meta.info[i].bbox.x2 - expected_res_x2), threshold_bbox);
        EXPECT_LT(abs(face_meta.info[i].bbox.y2 - expected_res_y2), threshold_bbox);
        EXPECT_LT(abs(face_meta.info[i].bbox.score - expected_res_bbox_conf), threshold_score);
        EXPECT_LT(abs(face_meta.info[i].hardhat_score - expected_res_hardhat_score),
                  threshold_score);
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai