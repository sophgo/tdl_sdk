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

class ThermalPersonDetectionTestSuite : public CVIAIModelTestSuite {
 public:
  ThermalPersonDetectionTestSuite()
      : CVIAIModelTestSuite("reg_daily_thermal_person_detection.json",
                            "reg_daily_thermal_person_detection") {}

  virtual ~ThermalPersonDetectionTestSuite() = default;

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

TEST_F(ThermalPersonDetectionTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    const char *model_path_get =
        CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
                 std::string(model_path_get));
  }
}

TEST_F(ThermalPersonDetectionTestSuite, get_vpss_config) {
  uint32_t dstWidth = 640;
  uint32_t dstHeight = 640;

  for (size_t test_index = 0; test_index < 1; test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, m_model_path.c_str(),
                           false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    cvai_vpssconfig_t vpssconfig;
    EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, 640, 640,
                                      0, &vpssconfig),
              CVIAI_SUCCESS);

    float factor[3] = {0.50098038, 0.50098038, 0.50098038};
    float mean[3] = {0.0, 0.0, 0.0};

    EXPECT_EQ(vpssconfig.chn_attr.u32Width, dstWidth);
    EXPECT_EQ(vpssconfig.chn_attr.u32Height, dstHeight);
    EXPECT_EQ(vpssconfig.chn_attr.enVideoFormat, VIDEO_FORMAT_LINEAR);
    EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_BGR_888_PLANAR);

    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.factor[i], factor[i]);
    }
    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.mean[i], mean[i]);
    }
  }
}

TEST_F(ThermalPersonDetectionTestSuite, skip_vpss_preprocess) {
  for (size_t test_index = 0; test_index < 1; test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    // select image_0 for test
    std::string image_path =
        (m_image_dir / std::string(m_json_object[test_index]["test_images"][0])).string();

    Image frame(image_path, PIXEL_FORMAT_BGR_888);

    ASSERT_TRUE(frame.open());

    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON,
                             m_model_path.c_str(), false);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      cvai_object_t obj;
      memset(&obj, 0, sizeof(cvai_object_t));
      EXPECT_EQ(CVI_AI_ThermalPerson(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS);
    }
    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON,
                             m_model_path.c_str(), true);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      cvai_object_t obj;
      memset(&obj, 0, sizeof(cvai_object_t));
      EXPECT_EQ(CVI_AI_ThermalPerson(m_ai_handle, frame.getFrame(), &obj), CVIAI_ERR_INFERENCE);
    }
  }
}

TEST_F(ThermalPersonDetectionTestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, m_model_path.c_str(),
                           false);
    CVI_AI_SetModelThreshold(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, 0.01);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    for (int img_idx = 0; img_idx < 1; img_idx++) {
      // select image_0 for test
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      {
        Image frame(image_path, PIXEL_FORMAT_BGR_888);
        ASSERT_TRUE(frame.open());

        cvai_object_t obj;
        memset(&obj, 0, sizeof(cvai_object_t));
        EXPECT_EQ(CVI_AI_ThermalPerson(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS);
#if 0
        for (uint32_t i = 0; i < obj.size; i++) {
          printf(
              "[%d][%d], x1, y1, x2, y2, result : [%f, %f, %f, %f], class : %d, %s\n",
              img_idx, i,
              obj.info[i].bbox.x1, obj.info[i].bbox.y1, 
              obj.info[i].bbox.x2, obj.info[i].bbox.y2,
              obj.info[i].classes,
              obj.info[i].name);
        }
#endif
      }
    }
  }
}

bool thermal_person_match_detections(cvai_object_t &obj_meta, cvai_bbox_t &expected_bbox,
                                     float iou_threshold, float score_threshold) {
  bool found = false;
  for (uint32_t actual_det_index = 0; actual_det_index < obj_meta.size; actual_det_index++) {
    found = iou(obj_meta.info[actual_det_index].bbox, expected_bbox) >= iou_threshold &&
            abs(obj_meta.info[actual_det_index].bbox.score - expected_bbox.score) < score_threshold;
    if (found) {
      // printf("found actual_det_index %d\n", actual_det_index);
      return found;
    }
  }
  return false;
}

TEST_F(ThermalPersonDetectionTestSuite, accruacy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, m_model_path.c_str(),
                           false);
    CVI_AI_SetModelThreshold(m_ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, 0.01);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    int img_num = int(m_json_object[test_index]["test_images"].size());
    float score_threshold = float(m_json_object[test_index]["score_threshold"]);
    float iou_threshold = float(m_json_object[test_index]["iou_threshold"]);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());

      cvai_object_t obj;
      memset(&obj, 0, sizeof(cvai_object_t));
      { EXPECT_EQ(CVI_AI_ThermalPerson(m_ai_handle, frame.getFrame(), &obj), CVIAI_SUCCESS); }

      for (uint32_t i = 0; i < obj.size; i++) {
#if 0
        printf(
            "[%d][%d], x1, y1, x2, y2, score, result : [%f, %f, %f, %f], %f, class : %d, %s\n",
            img_idx, i,
            obj.info[i].bbox.x1, obj.info[i].bbox.y1, 
            obj.info[i].bbox.x2, obj.info[i].bbox.y2,
            obj.info[i].bbox.score,
            obj.info[i].classes,
            obj.info[i].name);
#endif

        cvai_bbox_t expected_bbox = {
            .x1 = float(m_json_object[test_index]["expected_results"][img_idx][1][i][0]),
            .y1 = float(m_json_object[test_index]["expected_results"][img_idx][1][i][1]),
            .x2 = float(m_json_object[test_index]["expected_results"][img_idx][1][i][2]),
            .y2 = float(m_json_object[test_index]["expected_results"][img_idx][1][i][3]),
            .score = float(m_json_object[test_index]["expected_results"][img_idx][1][i][4]),
        };

        EXPECT_TRUE(
            thermal_person_match_detections(obj, expected_bbox, iou_threshold, score_threshold))
            << "image path: " << image_path << "\n"
            << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1 << ", "
            << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n";
      }
    }
  }
}

}  // namespace unitest
}  // namespace cviai