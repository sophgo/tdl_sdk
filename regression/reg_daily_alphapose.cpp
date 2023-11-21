#include <experimental/filesystem>
#include <fstream>
#include <memory>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "gtest.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class AlphaposeTestSuite : public CVI_TDLModelTestSuite {
 public:
  AlphaposeTestSuite() : CVI_TDLModelTestSuite("reg_daily_alphapose.json", "reg_daily_alphapose") {}

  virtual ~AlphaposeTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["pose_model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(AlphaposeTestSuite, open_close_model) {
  TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           false);
  ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

  const char *model_path_get =
      CVI_TDL_GetModelPath(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));
}

TEST_F(AlphaposeTestSuite, get_vpss_config) {
  TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                           false);
  ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
  cvtdl_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;

  EXPECT_EQ(CVI_TDL_GetVpssChnConfig(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, 100, 100, 0,
                                     &vpssconfig),
            CVI_TDL_ERR_GET_VPSS_CHN_CONFIG);

  EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);
}

TEST_F(AlphaposeTestSuite, skip_vpss_preprocess) {
  std::string image_path = (m_image_dir / std::string(m_json_object["test_images"][0])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvtdl_object_t obj;
  memset(&obj, 0, sizeof(cvtdl_object_t));
  // CVI_TDL_MobileDetV2_Pedestrian(tdl_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][0][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvtdl_object_info_t *)malloc(obj.size * sizeof(cvtdl_object_info_t));

  obj.info[0].bbox.x1 = float(m_json_object["expected_results"][0][1][0][0]);
  obj.info[0].bbox.y1 = float(m_json_object["expected_results"][0][1][0][1]);
  obj.info[0].bbox.x2 = float(m_json_object["expected_results"][0][1][0][2]);
  obj.info[0].bbox.y2 = float(m_json_object["expected_results"][0][1][0][3]);
  obj.info[0].classes = 0;

  {
    // test inference with skip vpss = false
    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                             false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
    ASSERT_EQ(CVI_TDL_AlphaPose(m_tdl_handle, frame.getFrame(), &obj), CVI_TDL_SUCCESS);
  }

  {
    // test inference with skip vpss = true
    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                             true);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
    ASSERT_EQ(CVI_TDL_AlphaPose(m_tdl_handle, frame.getFrame(), &obj), CVI_TDL_ERR_INVALID_ARGS);
  }
}

TEST_F(AlphaposeTestSuite, inference) {
  std::string image_path = (m_image_dir / std::string(m_json_object["test_images"][0])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvtdl_object_t obj;
  memset(&obj, 0, sizeof(cvtdl_object_t));
  // CVI_TDL_MobileDetV2_Pedestrian(tdl_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][0][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvtdl_object_info_t *)malloc(obj.size * sizeof(cvtdl_object_info_t));

  obj.info[0].bbox.x1 = float(m_json_object["expected_results"][0][1][0][0]);
  obj.info[0].bbox.y1 = float(m_json_object["expected_results"][0][1][0][1]);
  obj.info[0].bbox.x2 = float(m_json_object["expected_results"][0][1][0][2]);
  obj.info[0].bbox.y2 = float(m_json_object["expected_results"][0][1][0][3]);
  obj.info[0].classes = 0;

  {
    // test inference with skip vpss = false
    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                             false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
    ASSERT_EQ(CVI_TDL_AlphaPose(m_tdl_handle, frame.getFrame(), &obj), CVI_TDL_SUCCESS);
  }
}

TEST_F(AlphaposeTestSuite, accruacy) {
  float threshold = float(m_json_object["threshold"]);
  int img_idx = 0;
  std::string image_path =
      (m_image_dir / std::string(m_json_object["test_images"][img_idx])).string();

  Image frame(image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(frame.open());

  cvtdl_object_t obj;
  memset(&obj, 0, sizeof(cvtdl_object_t));
  // CVI_TDL_MobileDetV2_Pedestrian(tdl_handle, &frame, &obj);

  obj.size = int(m_json_object["expected_results"][img_idx][0]);
  obj.height = frame.getFrame()->stVFrame.u32Height;
  obj.width = frame.getFrame()->stVFrame.u32Width;
  obj.info = (cvtdl_object_info_t *)malloc(obj.size * sizeof(cvtdl_object_info_t));

  for (uint32_t i = 0; i < obj.size; i++) {
    obj.info[i].bbox.x1 = float(m_json_object["expected_results"][img_idx][1][i][0]);
    obj.info[i].bbox.y1 = float(m_json_object["expected_results"][img_idx][1][i][1]);
    obj.info[i].bbox.x2 = float(m_json_object["expected_results"][img_idx][1][i][2]);
    obj.info[i].bbox.y2 = float(m_json_object["expected_results"][img_idx][1][i][3]);
    obj.info[i].classes = 0;
  }

  {
    // test inference with skip vpss = true
    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_ALPHAPOSE, m_model_path.c_str(),
                             false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());
    ASSERT_EQ(CVI_TDL_AlphaPose(m_tdl_handle, frame.getFrame(), &obj), CVI_TDL_SUCCESS);

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
}  // namespace cvitdl