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

class AlphaposeTestSuite : public CVIAIModelTestSuite {
 public:
  AlphaposeTestSuite() : CVIAIModelTestSuite("reg_daily_alphapose.json", "reg_daily_alphapose") {}

  virtual ~AlphaposeTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["pose_model"]);
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

TEST_F(AlphaposeTestSuite, open_close_model) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                         false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));
}

TEST_F(AlphaposeTestSuite, get_vpss_config) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                         false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());
  cvai_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;

  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, 100, 100, 0,
                                    &vpssconfig),
            CVIAI_ERR_GET_VPSS_CHN_CONFIG);

  EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);
}

TEST_F(AlphaposeTestSuite, skip_vpss_preprocess) {
  std::string image_path = (m_image_dir / std::string(m_json_object["test_images"][0])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  // CVI_AI_MobileDetV2_Pedestrian(ai_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][0][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvai_object_info_t *)malloc(obj.size * sizeof(cvai_object_info_t));

  obj.info[0].bbox.x1 = float(m_json_object["expected_results"][0][1][0][0]);
  obj.info[0].bbox.y1 = float(m_json_object["expected_results"][0][1][0][1]);
  obj.info[0].bbox.x2 = float(m_json_object["expected_results"][0][1][0][2]);
  obj.info[0].bbox.y2 = float(m_json_object["expected_results"][0][1][0][3]);
  obj.info[0].classes = 0;

  {
    // test inference with skip vpss = false
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_AlphaPose(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS);
  }

  {
    // test inference with skip vpss = true
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           true);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_AlphaPose(m_ai_handle, frame.getFrame(), &obj), CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(AlphaposeTestSuite, inference) {
  std::string image_path = (m_image_dir / std::string(m_json_object["test_images"][0])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  // CVI_AI_MobileDetV2_Pedestrian(ai_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][0][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvai_object_info_t *)malloc(obj.size * sizeof(cvai_object_info_t));

  obj.info[0].bbox.x1 = float(m_json_object["expected_results"][0][1][0][0]);
  obj.info[0].bbox.y1 = float(m_json_object["expected_results"][0][1][0][1]);
  obj.info[0].bbox.x2 = float(m_json_object["expected_results"][0][1][0][2]);
  obj.info[0].bbox.y2 = float(m_json_object["expected_results"][0][1][0][3]);
  obj.info[0].classes = 0;

  {
    // test inference with skip vpss = false
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_AlphaPose(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS);
  }
}

TEST_F(AlphaposeTestSuite, accruacy) {
  float threshold = float(m_json_object["threshold"]);
  int img_idx = 0;
  std::string image_path =
      (m_image_dir / std::string(m_json_object["test_images"][img_idx])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  // CVI_AI_MobileDetV2_Pedestrian(ai_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][img_idx][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvai_object_info_t *)malloc(obj.size * sizeof(cvai_object_info_t));

  for (uint32_t i = 0; i < obj.size; i++) {
    obj.info[i].bbox.x1 = float(m_json_object["expected_results"][img_idx][1][i][0]);
    obj.info[i].bbox.y1 = float(m_json_object["expected_results"][img_idx][1][i][1]);
    obj.info[i].bbox.x2 = float(m_json_object["expected_results"][img_idx][1][i][2]);
    obj.info[i].bbox.y2 = float(m_json_object["expected_results"][img_idx][1][i][3]);
    obj.info[i].classes = 0;
  }

  {
    // test inference with skip vpss = true
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_AlphaPose(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS);

    for (uint32_t i = 0; i < obj.size; i++) {
      for (int point = 0; point < 17; point++) {
        float point_x = obj.info[i].pedestrian_properity->pose_17.x[point];
        float point_y = obj.info[i].pedestrian_properity->pose_17.y[point];

        float expected_res_x = float(m_json_object["expected_results"][img_idx][2][i][point][0]);
        float expected_res_y = float(m_json_object["expected_results"][img_idx][2][i][point][1]);
        // printf("point %d\n", i);
        // printf("point_x : %f \n", point_x);
        // printf("point_y : %f \n", point_y);
        // printf("expected_res_x : %f \n", expected_res_x);
        // printf("expected_res_y : %f \n", expected_res_y);
        EXPECT_LT(abs(point_x - expected_res_x), threshold);
        EXPECT_LT(abs(point_y - expected_res_y), threshold);
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai