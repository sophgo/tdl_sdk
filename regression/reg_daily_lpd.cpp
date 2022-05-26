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

#define DISTANCE_BIAS 12

static bool is_closed(cvai_4_pts_t *pts_1, cvai_4_pts_t *pts_2) {
  for (int i = 0; i < 4; i++) {
    if (ABS(pts_1->x[i] - pts_2->x[i]) > DISTANCE_BIAS) return false;
    if (ABS(pts_1->y[i] - pts_2->y[i]) > DISTANCE_BIAS) return false;
  }
  return true;
}

namespace fs = std::experimental::filesystem;
namespace cviai {
namespace unitest {

class LicensePlateDetectionTestSuite : public CVIAIModelTestSuite {
 public:
  LicensePlateDetectionTestSuite() : CVIAIModelTestSuite("daily_reg_LPD.json", "reg_daily_lpd") {}

  virtual ~LicensePlateDetectionTestSuite() = default;

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

TEST_F(LicensePlateDetectionTestSuite, open_close_model) {
  ASSERT_EQ(CVI_AI_OpenModel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, m_model_path.c_str()),
            CVIAI_SUCCESS)
      << "failed to set model path: " << m_model_path;

  const char *model_path_get = CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET);

  EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));

  ASSERT_EQ(CVI_AI_CloseModel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET), CVIAI_SUCCESS);
}

TEST_F(LicensePlateDetectionTestSuite, accruacy) {
  ASSERT_EQ(CVI_AI_OpenModel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, m_model_path.c_str()),
            CVIAI_SUCCESS);

  int img_num = int(m_json_object["reg_config"][0]["image_num"]);
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = std::string(m_json_object["reg_config"][0]["test_images"][img_idx]);
    image_path = (m_image_dir / image_path).string();

    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image_rgb.open());
    VIDEO_FRAME_INFO_S *vframe = image_rgb.getFrame();

    AIObject<cvai_object_t> vehicle_meta;
    init_obj_meta(vehicle_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);

    ASSERT_EQ(CVI_AI_LicensePlateDetection(m_ai_handle, vframe, vehicle_meta), CVIAI_SUCCESS);
    ASSERT_NE(vehicle_meta->info[0].vehicle_properity, (void *)0);
    cvai_4_pts_t *pred = &vehicle_meta->info[0].vehicle_properity->license_pts;
    cvai_4_pts_t *expected_res = new cvai_4_pts_t;
    for (int i = 0; i < 4; i++) {
      expected_res->x[i] =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx][2 * i]);
      expected_res->y[i] =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx][2 * i + 1]);
    }
    bool matched = is_closed(pred, expected_res);
    EXPECT_TRUE(matched);
    if (!matched) {
      printf("license plate: (%f,%f,%f,%f,%f,%f,%f,%f)\n", pred->x[0], pred->y[0], pred->x[1],
             pred->y[1], pred->x[2], pred->y[2], pred->x[3], pred->y[3]);
      printf("expected: (%f,%f,%f,%f,%f,%f,%f,%f)\n", expected_res->x[0], expected_res->y[0],
             expected_res->x[1], expected_res->y[1], expected_res->x[2], expected_res->y[2],
             expected_res->x[3], expected_res->y[3]);
    }

    delete expected_res;
  }
}

}  // namespace unitest
}  // namespace cviai