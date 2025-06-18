#include <fstream>
#include <string>
#include <unordered_map>

#include <gtest.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace cvitdl {
namespace unitest {

class MotionDetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  MotionDetectionTestSuite() : CVI_TDLModelTestSuite("daily_reg_md.json", "reg_daily_md") {}

  virtual ~MotionDetectionTestSuite() = default;

 protected:
  virtual void SetUp() {
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }

  float bbox_threshold = 0.95;
};

TEST_F(MotionDetectionTestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    uint8_t thresh = (uint8_t)m_json_object[test_index]["thresh"];
    float minarea = (float)m_json_object[test_index]["minarea"];

    std::string bgimg_path = (m_image_dir / m_json_object[test_index]["background"]).string();
    Image bg_image(bgimg_path, PIXEL_FORMAT_YUV_400);
    ASSERT_TRUE(bg_image.open());

    ASSERT_EQ(CVI_TDL_Set_MotionDetection_Background(m_tdl_handle, bg_image.getFrame()),
              CVI_TDL_SUCCESS);

    auto results = m_json_object[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();
      Image image(image_path, PIXEL_FORMAT_YUV_400);
      ASSERT_TRUE(image.open());

      TDLObject<cvtdl_object_t> obj_meta;

      ASSERT_EQ(CVI_TDL_MotionDetection(m_tdl_handle, image.getFrame(), obj_meta, thresh, minarea),
                CVI_TDL_SUCCESS);

      auto expected_dets = iter.value();

      ASSERT_EQ(obj_meta->size, expected_dets.size()) << "image path: " << image_path;

      bool missed = false;
      if (obj_meta->size != expected_dets.size()) {
        missed = true;
      }

      if (!missed) {
        for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
          auto bbox = expected_dets[det_index]["bbox"];

          cvtdl_bbox_t expected_bbox = {
              .x1 = float(bbox[0]),
              .y1 = float(bbox[1]),
              .x2 = float(bbox[2]),
              .y2 = float(bbox[3]),
          };

          auto comp = [=](cvtdl_object_info_t &info, cvtdl_bbox_t &bbox) {
            if (iou(info.bbox, bbox) >= bbox_threshold) {
              return true;
            }
            return false;
          };

          bool matched = match_dets(*obj_meta, expected_bbox, comp);

          ASSERT_TRUE(matched) << "image path: " << image_path << "\n"
                               << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1
                               << ", " << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n";
          if (!matched) {
            missed = true;
          }
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
}  // namespace cvitdl
