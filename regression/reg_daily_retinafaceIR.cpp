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

class RetinafaceIRTestSuite : public CVIAIModelTestSuite {
 public:
  RetinafaceIRTestSuite()
      : CVIAIModelTestSuite("reg_daily_retinafaceIR.json", "reg_daily_retinafaceIR") {}

  virtual ~RetinafaceIRTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    m_ai_handle = NULL;
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }
};

TEST_F(RetinafaceIRTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    const char *model_path_get =
        CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
                 std::string(model_path_get));
  }
}

TEST_F(RetinafaceIRTestSuite, get_vpss_config) {
  uint32_t dstWidth[3] = {608, 342, 608};
  uint32_t dstHeight[3] = {608, 608, 342};

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    cvai_vpssconfig_t vpssconfig;
    vpssconfig.chn_attr.u32Height = 200;
    vpssconfig.chn_attr.u32Width = 200;
    vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
    vpssconfig.chn_attr.stNormalize.bEnable = false;

    EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, 342, 608,
                                      0, &vpssconfig),
              CVIAI_SUCCESS);

    float factor[3] = {0.84685433, 0.84685433, 0.84685433};
    float mean[3] = {104.16309, 99.081955, 88.072853};

    uint32_t model_width = dstWidth[test_index];
    uint32_t model_height = dstHeight[test_index];

    EXPECT_EQ(vpssconfig.chn_attr.u32Width, model_width);
    EXPECT_EQ(vpssconfig.chn_attr.u32Height, model_height);
    EXPECT_EQ(vpssconfig.chn_attr.enVideoFormat, VIDEO_FORMAT_LINEAR);
    EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_BGR_888_PLANAR);

    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.factor[i], factor[i]);
    }
    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.mean[i], mean[i]);
    }
  }
}

TEST_F(RetinafaceIRTestSuite, skip_vpss_preprocess) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    // select image_0 for test
    std::string image_path =
        (m_image_dir / std::string(m_json_object[test_index]["test_images"][0])).string();

    Image frame(image_path, PIXEL_FORMAT_BGR_888);
    ASSERT_TRUE(frame.open());

    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR,
                             m_model_path.c_str(), false);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      cvai_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvai_face_t));
      EXPECT_EQ(CVI_AI_RetinaFace_IR(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS);
    }
    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR,
                             m_model_path.c_str(), true);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 1);
      EXPECT_EQ(CVI_AI_RetinaFace_IR(m_ai_handle, frame.getFrame(), face_meta),
                CVIAI_ERR_INFERENCE);
    }
  }
}

TEST_F(RetinafaceIRTestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    for (int img_idx = 0; img_idx < 1; img_idx++) {
      // select image_0 for test
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      {
        Image frame(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
        ASSERT_TRUE(frame.open());

        cvai_face_t face_meta;
        memset(&face_meta, 0, sizeof(cvai_face_t));
        EXPECT_EQ(CVI_AI_RetinaFace_IR(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS);
      }

      {
        Image frame(image_path, PIXEL_FORMAT_BGR_888);
        ASSERT_TRUE(frame.open());

        cvai_face_t face_meta;
        memset(&face_meta, 0, sizeof(cvai_face_t));
        EXPECT_EQ(CVI_AI_RetinaFace_IR(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS);
      }
    }
  }
}

TEST_F(RetinafaceIRTestSuite, accruacy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE_IR, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    int img_num = int(m_json_object[test_index]["test_images"].size());
    float threshold = float(m_json_object[test_index]["threshold"]);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());

      cvai_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvai_face_t));

      { EXPECT_EQ(CVI_AI_RetinaFace_IR(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS); }

      for (uint32_t i = 0; i < face_meta.size; i++) {
        float expected_res_x1 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][0]);
        float expected_res_y1 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][1]);
        float expected_res_x2 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][2]);
        float expected_res_y2 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][3]);
        {
          EXPECT_LT(abs(face_meta.info[i].bbox.x1 - expected_res_x1), threshold);
          EXPECT_LT(abs(face_meta.info[i].bbox.y1 - expected_res_y1), threshold);
          EXPECT_LT(abs(face_meta.info[i].bbox.x2 - expected_res_x2), threshold);
          EXPECT_LT(abs(face_meta.info[i].bbox.y2 - expected_res_y2), threshold);
        }
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai