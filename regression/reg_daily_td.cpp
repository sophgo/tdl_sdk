#include <experimental/filesystem>
#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "gtest.h"
#include "ive/ive.h"
#include "json.hpp"
#include "raii.hpp"

#define SCORE_BIAS 0.05

namespace fs = std::experimental::filesystem;
namespace cviai {
namespace unitest {

class TamperDetectionTestSuite : public CVIAIModelTestSuite {
 public:
  TamperDetectionTestSuite() : CVIAIModelTestSuite("daily_reg_TD.json", "reg_daily_td") {}

  virtual ~TamperDetectionTestSuite() = default;

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

TEST_F(TamperDetectionTestSuite, accruacy) {
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  int img_num = int(m_json_object["reg_config"][0]["image_num"]);
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = std::string(m_json_object["reg_config"][0]["test_images"][img_idx]);
    image_path = (m_image_dir / image_path).string();
    float expected_res = float(m_json_object["reg_config"][0]["expected_results"][img_idx]);

    IVE_IMAGE_S ive_frame =
        CVI_IVE_ReadImage(ive_handle, image_path.c_str(), IVE_IMAGE_TYPE_U8C3_PLANAR);
    if (ive_frame.u16Width == 0) {
      FAIL() << "Read image failed!\n";
    }

    // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
    VIDEO_FRAME_INFO_S frame;
    ASSERT_EQ(CVI_IVE_Image2VideoFrameInfo(&ive_frame, &frame, false), CVI_SUCCESS);

    float moving_score;
    ASSERT_EQ(CVI_AI_TamperDetection(m_ai_handle, &frame, &moving_score), CVIAI_SUCCESS);
    // printf("[%d] %f (expected: %f)\n", img_idx, moving_score, expected_res);

    ASSERT_LT(ABS(moving_score - expected_res), SCORE_BIAS);
    CVI_SYS_FreeI(ive_handle, &ive_frame);
  }
  CVI_IVE_DestroyHandle(ive_handle);
}

}  // namespace unitest
}  // namespace cviai