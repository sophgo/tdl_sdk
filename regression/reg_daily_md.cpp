#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_test.hpp"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace cviai {
namespace unitest {

class MotionDetectionTestSuite : public CVIAIModelTestSuite {
 public:
  MotionDetectionTestSuite() : CVIAIModelTestSuite("daily_reg_md.json", "reg_daily_md") {}

  virtual ~MotionDetectionTestSuite() = default;

 protected:
  virtual void SetUp() {
    ASSERT_EQ(CVI_AI_CreateHandle2(&m_ai_handle, 1, 0), CVIAI_SUCCESS);
    ASSERT_EQ(CVI_AI_SetVpssTimeout(m_ai_handle, 1000), CVIAI_SUCCESS);
  }

  virtual void TearDown() {
    CVI_AI_DestroyHandle(m_ai_handle);
    m_ai_handle = NULL;
  }
};

bool match_detections(cvai_object_t *obj_meta, cvai_bbox_t &expected_bbox, float bbox_threhold) {
  bool found = false;
  for (uint32_t actual_det_index = 0; actual_det_index < obj_meta->size; actual_det_index++) {
    found = iou(obj_meta->info[actual_det_index].bbox, expected_bbox) >= bbox_threhold;
    if (found) {
      return found;
    }
  }

  return false;
}

TEST_F(MotionDetectionTestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    uint8_t thresh = (uint8_t)m_json_object[test_index]["thresh"];
    float minarea = (float)m_json_object[test_index]["minarea"];

    std::string bgimg_path = (m_image_dir / m_json_object[test_index]["background"]).string();
    Image bg_image(bgimg_path, PIXEL_FORMAT_YUV_400);
    ASSERT_TRUE(bg_image.open());

    ASSERT_EQ(CVI_AI_Set_MotionDetection_Background(m_ai_handle, bg_image.getFrame()),
              CVIAI_SUCCESS);

    auto results = m_json_object[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();
      Image image(image_path, PIXEL_FORMAT_YUV_400);
      ASSERT_TRUE(image.open());

      AIObject<cvai_object_t> obj_meta;

      ASSERT_EQ(CVI_AI_MotionDetection(m_ai_handle, image.getFrame(), obj_meta, thresh, minarea),
                CVIAI_SUCCESS);

      auto expected_dets = iter.value();

      EXPECT_EQ(obj_meta->size, expected_dets.size()) << "image path: " << image_path;

      if (obj_meta->size != expected_dets.size()) {
        continue;
      }

      bool missed = false;
      for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];

        cvai_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]),
            .y2 = float(bbox[3]),
        };

        bool matched = match_detections(obj_meta, expected_bbox, 0.95);

        EXPECT_TRUE(matched) << "image path: " << image_path << "\n"
                             << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1
                             << ", " << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n";
        if (!matched) {
          missed = true;
        }
      }

      if (missed) {
        for (uint32_t pred_idx = 0; pred_idx < obj_meta->size; pred_idx++) {
          printf("actual det[%d] = {%f, %f, %f, %f}\n", pred_idx, obj_meta->info[pred_idx].bbox.x1,
                 obj_meta->info[pred_idx].bbox.y1, obj_meta->info[pred_idx].bbox.x2,
                 obj_meta->info[pred_idx].bbox.y2);
        }
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai
