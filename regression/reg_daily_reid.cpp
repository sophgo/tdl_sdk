#include <experimental/filesystem>
#include <fstream>
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

#define DISTANCE_BIAS 0.05

static float cosine_distance(const cvtdl_feature_t &features1, const cvtdl_feature_t &features2) {
  int32_t value1 = 0, value2 = 0, value3 = 0;
  for (uint32_t i = 0; i < features1.size; i++) {
    value1 += (short)features1.ptr[i] * features1.ptr[i];
    value2 += (short)features2.ptr[i] * features2.ptr[i];
    value3 += (short)features1.ptr[i] * features2.ptr[i];
  }
  return 1 - ((float)value3 / (sqrt((double)value1) * sqrt((double)value2)));
}

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class ReIdentificationTestSuite : public CVI_TDLModelTestSuite {
 public:
  ReIdentificationTestSuite() : CVI_TDLModelTestSuite("daily_reg_ReID.json", "reg_daily_reid") {}

  virtual ~ReIdentificationTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["reg_config"][0]["model_name"]);
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

TEST_F(ReIdentificationTestSuite, open_close_model) {
  ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, m_model_path.c_str()),
            CVI_TDL_SUCCESS)
      << "failed to set model path: " << m_model_path;

  const char *model_path_get = CVI_TDL_GetModelPath(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET);

  EXPECT_PRED2([](const std::string &s1, const std::string &s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));

  ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET), CVI_TDL_SUCCESS);
}

TEST_F(ReIdentificationTestSuite, accruacy) {
  ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, m_model_path.c_str()),
            CVI_TDL_SUCCESS);

  int img_num = int(m_json_object["reg_config"][0]["image_num"]);
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    TDLObject<cvtdl_object_t> pedestrian_meta[2];
    for (int i = 0; i < 2; i++) {
      std::string image_path =
          std::string(m_json_object["reg_config"][0]["test_images"][img_idx][i]);
      image_path = (m_image_dir / image_path).string();

      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image_rgb.open());
      VIDEO_FRAME_INFO_S *vframe = image_rgb.getFrame();

      init_obj_meta(pedestrian_meta[i], 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width,
                    0);

      ASSERT_EQ(CVI_TDL_OSNet(m_tdl_handle, vframe, pedestrian_meta[i]), CVI_TDL_SUCCESS);
    }

    float distance =
        cosine_distance(pedestrian_meta[0]->info[0].feature, pedestrian_meta[1]->info[0].feature);
    float expected_distance = float(m_json_object["reg_config"][0]["expected_results"][img_idx]);
    // printf("cos distance: %f (expected: %f)\n", distance, expected_distance);

    ASSERT_LT(ABS(distance - expected_distance), DISTANCE_BIAS);
    CVI_TDL_FreeCpp(pedestrian_meta[0]);
    CVI_TDL_FreeCpp(pedestrian_meta[1]);
  }
}

}  // namespace unitest
}  // namespace cvitdl
