#include <experimental/filesystem>
#include <fstream>
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

#define FORMAT_NUM 2

typedef enum { taiwan = 0, china } LicenseFormat;

namespace fs = std::experimental::filesystem;
namespace cviai {
namespace unitest {

class LicensePlateRecognitionTestSuite : public CVIAIModelTestSuite {
 public:
  LicensePlateRecognitionTestSuite() : CVIAIModelTestSuite("daily_reg_LPR.json", "reg_daily_lpr") {}

  virtual ~LicensePlateRecognitionTestSuite() = default;
  struct ModelInfo {
    std::string abbreviation;
    CVI_AI_SUPPORTED_MODEL_E idx;
    std::string path;
  };

  /* make sure FORMAT_NUM match the length of m_model_path */
  const LicenseFormat FORMATS[FORMAT_NUM] = {taiwan, china};
  std::map<LicenseFormat, ModelInfo> m_model_info = {
      {taiwan, {"TW", CVI_AI_SUPPORTED_MODEL_LPRNET_TW, ""}},
      {china, {"CN", CVI_AI_SUPPORTED_MODEL_LPRNET_CN, ""}}};

 protected:
  virtual void SetUp() {
    for (int i = 0; i < FORMAT_NUM; i++) {
      std::string f = m_model_info[FORMATS[i]].abbreviation;
      std::string m_name = std::string(m_json_object[f.c_str()]["reg_config"][0]["model_name"]);
      m_model_info[FORMATS[i]].path = (m_model_dir / fs::path(m_name)).string();
    }

    m_ai_handle = NULL;
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }
};

TEST_F(LicensePlateRecognitionTestSuite, open_close_model) {
  for (int i = 0; i < FORMAT_NUM; i++) {
    ModelInfo &m_info = m_model_info[FORMATS[i]];
    ASSERT_LT(m_info.idx, CVI_AI_SUPPORTED_MODEL_END);

    ASSERT_EQ(CVI_AI_OpenModel(m_ai_handle, m_info.idx, m_info.path.c_str()), CVIAI_SUCCESS)
        << "failed to set model path: " << m_info.path;

    const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, m_info.idx);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_info.path,
                 std::string(model_path_get));

    ASSERT_EQ(CVI_AI_CloseModel(m_ai_handle, m_info.idx), CVIAI_SUCCESS);
  }
}

TEST_F(LicensePlateRecognitionTestSuite, accruacy) {
  for (int i = 0; i < FORMAT_NUM; i++) {
    ModelInfo &m_info = m_model_info[FORMATS[i]];
    ASSERT_EQ(CVI_AI_OpenModel(m_ai_handle, m_info.idx, m_info.path.c_str()), CVIAI_SUCCESS);
    int img_num = int(m_json_object[m_info.abbreviation]["reg_config"][0]["image_num"]);
    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      std::string image_path =
          std::string(m_json_object[m_info.abbreviation]["reg_config"][0]["test_images"][img_idx]);
      image_path = (m_image_dir / image_path).string();
      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image_rgb.open());
      VIDEO_FRAME_INFO_S *vframe = image_rgb.getFrame();

      AIObject<cvai_object_t> vehicle_meta;
      init_obj_meta(vehicle_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);
      init_vehicle_meta(vehicle_meta);
      switch (FORMATS[i]) {
        case taiwan:
          CVI_AI_LicensePlateRecognition_TW(m_ai_handle, vframe, vehicle_meta);
          break;
        case china:
          CVI_AI_LicensePlateRecognition_CN(m_ai_handle, vframe, vehicle_meta);
          break;
        default:
          FAIL() << "Shouldn't get here.";
      }

      std::string expected_res = std::string(
          m_json_object[m_info.abbreviation]["reg_config"][0]["expected_results"][img_idx]);
#if 0
      printf("ID number: %s (expected: %s)\n",
             vehicle_meta->info[0].vehicle_properity->license_char, expected_res.c_str());
#endif
      ASSERT_EQ(strcmp(vehicle_meta->info[0].vehicle_properity->license_char, expected_res.c_str()),
                0);
    }

    ASSERT_EQ(CVI_AI_CloseModel(m_ai_handle, m_info.idx), CVIAI_SUCCESS);
  }
}

}  // namespace unitest
}  // namespace cviai