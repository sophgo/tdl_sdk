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

class LivenessTestSuite : public CVIAIModelTestSuite {
 public:
  LivenessTestSuite() : CVIAIModelTestSuite("reg_daily_liveness.json", "reg_daily_liveness") {}

  virtual ~LivenessTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["liveness_model"]);
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

TEST_F(LivenessTestSuite, open_close_model) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));
}

TEST_F(LivenessTestSuite, get_vpss_config) {
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());
  cvai_vpssconfig_t vpssconfig;
  vpssconfig.chn_attr.u32Height = 200;
  vpssconfig.chn_attr.u32Width = 200;
  vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
  vpssconfig.chn_attr.stNormalize.bEnable = false;

  EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, 100, 100, 0,
                                    &vpssconfig),
            CVIAI_ERR_GET_VPSS_CHN_CONFIG);

  EXPECT_EQ(vpssconfig.chn_attr.u32Height, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.u32Width, (uint32_t)200);
  EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_ARGB_1555);
  EXPECT_EQ(vpssconfig.chn_attr.stNormalize.bEnable, false);
}

TEST_F(LivenessTestSuite, skip_vpss_preprocess) {
  std::string rgb_image_path =
      (m_image_dir / std::string(m_json_object["test_images"][0][0])).string();
  std::string ir_image_path =
      (m_image_dir / std::string(m_json_object["test_images"][0][1])).string();

  Image rgb_image(rgb_image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(rgb_image.open());
  Image ir_image(ir_image_path, PIXEL_FORMAT_BGR_888);
  ASSERT_TRUE(ir_image.open());

  cvai_face_t rgb_face;
  memset(&rgb_face, 0, sizeof(cvai_face_t));
  cvai_face_t ir_face;
  memset(&ir_face, 0, sizeof(cvai_face_t));

  rgb_face.size = 1;
  rgb_face.width = rgb_image.getFrame()->stVFrame.u32Width;
  rgb_face.height = rgb_image.getFrame()->stVFrame.u32Height;
  rgb_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * rgb_face.size);
  memset(rgb_face.info, 0, sizeof(cvai_face_info_t) * rgb_face.size);

  ir_face.size = 1;
  ir_face.width = ir_image.getFrame()->stVFrame.u32Width;
  ir_face.height = ir_image.getFrame()->stVFrame.u32Height;
  ir_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * ir_face.size);
  memset(ir_face.info, 0, sizeof(cvai_face_info_t) * ir_face.size);

  rgb_face.info[0].bbox.x1 = float(m_json_object["bboxs"][0][0][0]);
  rgb_face.info[0].bbox.y1 = float(m_json_object["bboxs"][0][0][1]);
  rgb_face.info[0].bbox.x2 = float(m_json_object["bboxs"][0][0][2]);
  rgb_face.info[0].bbox.y2 = float(m_json_object["bboxs"][0][0][3]);

  ir_face.info[0].bbox.x1 = float(m_json_object["bboxs"][0][1][0]);
  ir_face.info[0].bbox.y1 = float(m_json_object["bboxs"][0][1][1]);
  ir_face.info[0].bbox.x2 = float(m_json_object["bboxs"][0][1][2]);
  ir_face.info[0].bbox.y2 = float(m_json_object["bboxs"][0][1][3]);

  {
    // test inference with skip vpss = false
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_Liveness(m_ai_handle, rgb_image.getFrame(), ir_image.getFrame(), &rgb_face,
                              &ir_face),
              CVIAI_SUCCESS);
  }

  {
    // test inference with skip vpss = true
    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(),
                           true);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    ASSERT_EQ(CVI_AI_Liveness(m_ai_handle, rgb_image.getFrame(), ir_image.getFrame(), &rgb_face,
                              &ir_face),
              CVIAI_ERR_INVALID_ARGS);
  }
}

TEST_F(LivenessTestSuite, inference) {
  // test inference with skip vpss = false
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  // select image_0 for test
  std::string rgb_image_path =
      (m_image_dir / std::string(m_json_object["test_images"][0][0])).string();
  std::string ir_image_path =
      (m_image_dir / std::string(m_json_object["test_images"][0][1])).string();

  cvai_face_t rgb_face;
  memset(&rgb_face, 0, sizeof(cvai_face_t));
  cvai_face_t ir_face;
  memset(&ir_face, 0, sizeof(cvai_face_t));

  rgb_face.size = 1;
  rgb_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * rgb_face.size);
  memset(rgb_face.info, 0, sizeof(cvai_face_info_t) * rgb_face.size);

  ir_face.size = 1;
  ir_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * ir_face.size);
  memset(ir_face.info, 0, sizeof(cvai_face_info_t) * ir_face.size);

  rgb_face.info[0].bbox.x1 = float(m_json_object["bboxs"][0][0][0]);
  rgb_face.info[0].bbox.y1 = float(m_json_object["bboxs"][0][0][1]);
  rgb_face.info[0].bbox.x2 = float(m_json_object["bboxs"][0][0][2]);
  rgb_face.info[0].bbox.y2 = float(m_json_object["bboxs"][0][0][3]);

  ir_face.info[0].bbox.x1 = float(m_json_object["bboxs"][0][1][0]);
  ir_face.info[0].bbox.y1 = float(m_json_object["bboxs"][0][1][1]);
  ir_face.info[0].bbox.x2 = float(m_json_object["bboxs"][0][1][2]);
  ir_face.info[0].bbox.y2 = float(m_json_object["bboxs"][0][1][3]);

  {
    // test inference with PIXEL_FORMAT_BGR_888
    Image rgb_image(rgb_image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(rgb_image.open());
    Image ir_image(ir_image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(ir_image.open());

    ASSERT_EQ(CVI_AI_Liveness(m_ai_handle, rgb_image.getFrame(), ir_image.getFrame(), &rgb_face,
                              &ir_face),
              CVIAI_SUCCESS);
  }

  {
    // test inference with PIXEL_FORMAT_RGB_888_PLANAR
    Image rgb_image(rgb_image_path, PIXEL_FORMAT_RGB_888_PLANAR);
    ASSERT_TRUE(rgb_image.open());
    Image ir_image(ir_image_path, PIXEL_FORMAT_RGB_888_PLANAR);
    ASSERT_TRUE(ir_image.open());

    ASSERT_EQ(CVI_AI_Liveness(m_ai_handle, rgb_image.getFrame(), ir_image.getFrame(), &rgb_face,
                              &ir_face),
              CVIAI_SUCCESS);
  }
}

TEST_F(LivenessTestSuite, accruacy) {
  // test inference with skip vpss = false
  AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, m_model_path.c_str(), false);
  ASSERT_NO_FATAL_FAILURE(aimodel.open());

  int img_num = int(m_json_object["image_num"]);
  float threshold = float(m_json_object["threshold"]);

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    // select image_0 for test
    std::string rgb_image_path =
        (m_image_dir / std::string(m_json_object["test_images"][img_idx][0])).string();
    std::string ir_image_path =
        (m_image_dir / std::string(m_json_object["test_images"][img_idx][1])).string();
    float expected_res = float(m_json_object["expected_results"][img_idx]);

    // test inference with PIXEL_FORMAT_BGR_888
    Image rgb_image(rgb_image_path, PIXEL_FORMAT_BGR_888);
    ASSERT_TRUE(rgb_image.open());
    Image ir_image(ir_image_path, PIXEL_FORMAT_BGR_888);
    ASSERT_TRUE(ir_image.open());

    cvai_face_t rgb_face;
    memset(&rgb_face, 0, sizeof(cvai_face_t));
    cvai_face_t ir_face;
    memset(&ir_face, 0, sizeof(cvai_face_t));

    rgb_face.size = 1;
    rgb_face.width = rgb_image.getFrame()->stVFrame.u32Width;
    rgb_face.height = rgb_image.getFrame()->stVFrame.u32Height;
    rgb_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * rgb_face.size);
    memset(rgb_face.info, 0, sizeof(cvai_face_info_t) * rgb_face.size);

    ir_face.size = 1;
    ir_face.width = ir_image.getFrame()->stVFrame.u32Width;
    ir_face.height = ir_image.getFrame()->stVFrame.u32Height;
    ir_face.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * ir_face.size);
    memset(ir_face.info, 0, sizeof(cvai_face_info_t) * ir_face.size);

    rgb_face.info[0].bbox.x1 = float(m_json_object["bboxs"][img_idx][0][0]);
    rgb_face.info[0].bbox.y1 = float(m_json_object["bboxs"][img_idx][0][1]);
    rgb_face.info[0].bbox.x2 = float(m_json_object["bboxs"][img_idx][0][2]);
    rgb_face.info[0].bbox.y2 = float(m_json_object["bboxs"][img_idx][0][3]);

    ir_face.info[0].bbox.x1 = float(m_json_object["bboxs"][img_idx][1][0]);
    ir_face.info[0].bbox.y1 = float(m_json_object["bboxs"][img_idx][1][1]);
    ir_face.info[0].bbox.x2 = float(m_json_object["bboxs"][img_idx][1][2]);
    ir_face.info[0].bbox.y2 = float(m_json_object["bboxs"][img_idx][1][3]);

    {
      ASSERT_EQ(CVI_AI_Liveness(m_ai_handle, rgb_image.getFrame(), ir_image.getFrame(), &rgb_face,
                                &ir_face),
                CVIAI_SUCCESS);
      // printf("[%d] expected: %f, score: %f \n", img_idx, expected_res,
      //       rgb_face.info[0].liveness_score);
      EXPECT_LT(abs(rgb_face.info[0].liveness_score - expected_res), threshold);
    }
    CVI_AI_FreeCpp(&rgb_face);
    CVI_AI_FreeCpp(&ir_face);
  }
}

}  // namespace unitest
}  // namespace cviai
