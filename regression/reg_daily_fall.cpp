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

class FallTestSuite : public CVIAIModelTestSuite {
 public:
  FallTestSuite() : CVIAIModelTestSuite("reg_daily_fall.json", "reg_daily_fall") {}

  virtual ~FallTestSuite() = default;

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

TEST_F(FallTestSuite, accruacy) {
  int img_num = int(m_json_object["test_images"].size());

  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    int expected_res = int(m_json_object["expected_results"][img_idx]);
    std::string image_path =
        (m_image_dir / std::string(m_json_object["test_images"][img_idx])).string();

    cvai_object_t obj;
    memset(&obj, 0, sizeof(cvai_object_t));

    obj.size = 1;
    obj.height = 180;
    obj.width = 320;
    obj.info = (cvai_object_info_t *)malloc(obj.size * sizeof(cvai_object_info_t));
    obj.info[0].classes = 0;
    obj.info[0].pedestrian_properity = (cvai_pedestrian_meta *)malloc(sizeof(cvai_pedestrian_meta));

    if (m_json_object["points"][img_idx].size() > 0) {
      obj.info[0].bbox.x1 = float(m_json_object["bboxs"][img_idx][0]);
      obj.info[0].bbox.y1 = float(m_json_object["bboxs"][img_idx][1]);
      obj.info[0].bbox.x2 = float(m_json_object["bboxs"][img_idx][2]);
      obj.info[0].bbox.y2 = float(m_json_object["bboxs"][img_idx][3]);

      for (int point = 0; point < 17; point++) {
        obj.info[0].pedestrian_properity->pose_17.x[point] =
            float(m_json_object["points"][img_idx][point][0]);
        obj.info[0].pedestrian_properity->pose_17.y[point] =
            float(m_json_object["points"][img_idx][point][1]);
      }

      CVI_AI_Fall(m_ai_handle, &obj);
      if (obj.info[0].pedestrian_properity != NULL) {
        bool pass = expected_res == (int)obj.info[0].pedestrian_properity->fall;
        if (!pass) {
          printf("[%d] pass: %d; fall, expected : %d, result : %d\n", img_idx, pass, expected_res,
                 obj.info[0].pedestrian_properity->fall);
        }
        EXPECT_EQ(expected_res, (int)obj.info[0].pedestrian_properity->fall);
      }
    }
  }
}
}  // namespace unitest
}  // namespace cviai