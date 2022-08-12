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

class FaceMaskDetectionTestSuite : public CVIAIModelTestSuite {
 public:
  FaceMaskDetectionTestSuite() : CVIAIModelTestSuite("reg_daily_fdmask.json", "reg_daily_fdmask") {}

  virtual ~FaceMaskDetectionTestSuite() = default;

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

TEST_F(FaceMaskDetectionTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    const char *model_path_get =
        CVI_AI_GetModelPath(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, m_model_path,
                 std::string(model_path_get));
  }
}

TEST_F(FaceMaskDetectionTestSuite, get_vpss_config) {
  uint32_t dstWidth[3] = {768};
  uint32_t dstHeight[3] = {432};

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());
    cvai_vpssconfig_t vpssconfig;
    vpssconfig.chn_attr.u32Height = 200;
    vpssconfig.chn_attr.u32Width = 200;
    vpssconfig.chn_attr.enPixelFormat = PIXEL_FORMAT_ARGB_1555;
    vpssconfig.chn_attr.stNormalize.bEnable = false;

    EXPECT_EQ(CVI_AI_GetVpssChnConfig(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION, 342,
                                      608, 0, &vpssconfig),
              CVIAI_SUCCESS);

    float factor[3] = {0.8254323, 0.84385717, 0.84010673};
    float mean[3] = {102.48568, 98.508514, 87.317329};

    uint32_t model_width = dstWidth[test_index];
    uint32_t model_height = dstHeight[test_index];

    EXPECT_EQ(vpssconfig.chn_attr.u32Width, model_width);
    EXPECT_EQ(vpssconfig.chn_attr.u32Height, model_height);
    EXPECT_EQ(vpssconfig.chn_attr.enVideoFormat, VIDEO_FORMAT_LINEAR);
    EXPECT_EQ(vpssconfig.chn_attr.enPixelFormat, PIXEL_FORMAT_RGB_888_PLANAR);

    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.factor[i], factor[i]);
    }
    for (uint32_t i = 0; i < 3; i++) {
      EXPECT_FLOAT_EQ(vpssconfig.chn_attr.stNormalize.mean[i], mean[i]);
    }
  }
}

TEST_F(FaceMaskDetectionTestSuite, skip_vpss_preprocess) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    // select image_0 for test
    std::string image_path =
        (m_image_dir / std::string(m_json_object[test_index]["test_images"][0])).string();

    Image frame(image_path, PIXEL_FORMAT_BGR_888);
    ASSERT_TRUE(frame.open());

    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                             m_model_path.c_str(), false);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      cvai_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvai_face_t));
      EXPECT_EQ(CVI_AI_FaceMaskDetection(m_ai_handle, frame.getFrame(), &face_meta), CVIAI_SUCCESS);
    }
    {
      AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                             m_model_path.c_str(), true);
      ASSERT_NO_FATAL_FAILURE(aimodel.open());
      AIObject<cvai_face_t> face_meta;
      init_face_meta(face_meta, 1);
      EXPECT_EQ(CVI_AI_FaceMaskDetection(m_ai_handle, frame.getFrame(), face_meta),
                CVIAI_ERR_INFERENCE);
    }
  }
}

TEST_F(FaceMaskDetectionTestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    for (int img_idx = 0; img_idx < 1; img_idx++) {
      // select image_0 for test
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      {
        Image frame(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
        ASSERT_TRUE(frame.open());

        cvai_face_t face_meta;
        memset(&face_meta, 0, sizeof(cvai_face_t));
        EXPECT_EQ(CVI_AI_FaceMaskDetection(m_ai_handle, frame.getFrame(), &face_meta),
                  CVIAI_SUCCESS);
      }

      {
        Image frame(image_path, PIXEL_FORMAT_BGR_888);
        ASSERT_TRUE(frame.open());

        cvai_face_t face_meta;
        memset(&face_meta, 0, sizeof(cvai_face_t));
        EXPECT_EQ(CVI_AI_FaceMaskDetection(m_ai_handle, frame.getFrame(), &face_meta),
                  CVIAI_SUCCESS);
      }
    }
  }
}

TEST_F(FaceMaskDetectionTestSuite, accruacy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    AIModelHandler aimodel(m_ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION,
                           m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(aimodel.open());

    int img_num = int(m_json_object[test_index]["test_images"].size());
    float iou_threshold = float(m_json_object[test_index]["threshold"]);
    float score_threshold = float(m_json_object[test_index]["threshold_score"]);

    for (int img_idx = 0; img_idx < img_num; img_idx++) {
      std::string image_path =
          (m_image_dir / std::string(m_json_object[test_index]["test_images"][img_idx])).string();

      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());

      AIObject<cvai_face_t> face_meta;

      {
        EXPECT_EQ(CVI_AI_FaceMaskDetection(m_ai_handle, frame.getFrame(), face_meta),
                  CVIAI_SUCCESS);
      }

      for (uint32_t i = 0; i < face_meta->size; i++) {
        float expected_res_x1 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][0]);
        float expected_res_y1 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][1]);
        float expected_res_x2 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][2]);
        float expected_res_y2 =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][3]);
        float expected_res_mask_score =
            float(m_json_object[test_index]["expected_results"][img_idx][1][i][4]);

        cvai_face_info_t expected_faceinfo = {0};
        expected_faceinfo.bbox.x1 = expected_res_x1;
        expected_faceinfo.bbox.y1 = expected_res_y1;
        expected_faceinfo.bbox.x2 = expected_res_x2;
        expected_faceinfo.bbox.y2 = expected_res_y2;
        expected_faceinfo.mask_score = expected_res_mask_score;

        auto comp = [=](cvai_face_info_t &pred, cvai_face_info_t &expected) {
          if (iou(pred.bbox, expected.bbox) >= iou_threshold &&
              abs(pred.mask_score - expected.mask_score) < score_threshold) {
            return true;
          }
          return false;
        };

        bool matched = match_dets(*face_meta, expected_faceinfo, comp);
        EXPECT_TRUE(matched) << "image path: " << image_path << "\n"
                             << "model path: " << m_model_path << "\n"
                             << "expected bbox: (" << expected_faceinfo.bbox.x1 << ", "
                             << expected_faceinfo.bbox.y1 << ", " << expected_faceinfo.bbox.x2
                             << ", " << expected_faceinfo.bbox.y2 << ")\n";
      }
      CVI_AI_FreeCpp(face_meta);
    }
  }
}

}  // namespace unitest
}  // namespace cviai
