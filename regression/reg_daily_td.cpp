#include <experimental/filesystem>
#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_test.hpp"
#include "gtest.h"
#include "json.hpp"
#include "raii.hpp"

#define SCORE_BIAS 0.05

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class TamperDetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  TamperDetectionTestSuite() : CVI_TDLModelTestSuite("daily_reg_TD.json", "reg_daily_td") {}

  virtual ~TamperDetectionTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(TamperDetectionTestSuite, accruacy) {
  int img_num = int(m_json_object["reg_config"][0]["image_num"]);
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = std::string(m_json_object["reg_config"][0]["test_images"][img_idx]);
    image_path = (m_image_dir / image_path).string();
    float expected_res = float(m_json_object["reg_config"][0]["expected_results"][img_idx]);

    Image image(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
    ASSERT_TRUE(image.open());

    float moving_score;
    ASSERT_EQ(CVI_TDL_TamperDetection(m_tdl_handle, image.getFrame(), &moving_score),
              CVI_TDL_SUCCESS);
    // printf("[%d] %f (expected: %f)\n", img_idx, moving_score, expected_res);

    ASSERT_LT(ABS(moving_score - expected_res), SCORE_BIAS);
  }
}

}  // namespace unitest
}  // namespace cvitdl