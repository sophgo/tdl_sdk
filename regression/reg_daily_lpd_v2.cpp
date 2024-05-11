#include <gtest.h>
#include <fstream>
#include <string>
#include <unordered_map>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "json.hpp"
#include "raii.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class LicensePlateDetectionV2TestSuite : public CVI_TDLModelTestSuite {
 public:
  LicensePlateDetectionV2TestSuite()
      : CVI_TDLModelTestSuite("reg_daily_licenseplate.json", "reg_daily_lpd") {}

  virtual ~LicensePlateDetectionV2TestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(LicensePlateDetectionV2TestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    ASSERT_EQ(
        CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_DETECTION, m_model_path.c_str()),
        CVI_TDL_SUCCESS)
        << "failed to set model path: " << m_model_path.c_str();
    ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_DETECTION),
              CVI_TDL_SUCCESS);
  }
}

TEST_F(LicensePlateDetectionV2TestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_DETECTION,
                             m_model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    auto results = m_json_object[test_index]["results"];

    std::string image_path = (m_image_dir / results.begin().key()).string();
    {
      Image frame(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
      ASSERT_TRUE(frame.open());
      VIDEO_FRAME_INFO_S *vframe = frame.getFrame();
      TDLObject<cvtdl_object_t> obj_meta;
      init_obj_meta(obj_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);
      EXPECT_EQ(CVI_TDL_License_Plate_Detectionv2(m_tdl_handle, vframe, obj_meta), CVI_TDL_SUCCESS);
    }

    {
      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());
      VIDEO_FRAME_INFO_S *vframe = frame.getFrame();
      TDLObject<cvtdl_object_t> obj_meta;
      init_obj_meta(obj_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);
      EXPECT_EQ(CVI_TDL_License_Plate_Detectionv2(m_tdl_handle, vframe, obj_meta), CVI_TDL_SUCCESS);
    }
  }
}

TEST_F(LicensePlateDetectionV2TestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    ASSERT_EQ(
        CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_DETECTION, m_model_path.c_str()),
        CVI_TDL_SUCCESS);
    const float bbox_threshold = 0.90;
    const float score_threshold = 0.1;
    auto results = m_json_object[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();
      Image frame(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(frame.open());

      VIDEO_FRAME_INFO_S *vframe = frame.getFrame();
      TDLObject<cvtdl_object_t> obj_meta;
      init_obj_meta(obj_meta, 1, vframe->stVFrame.u32Height, vframe->stVFrame.u32Width, 0);

      ASSERT_EQ(CVI_TDL_License_Plate_Detectionv2(m_tdl_handle, vframe, obj_meta), CVI_TDL_SUCCESS);

      auto expected_dets = iter.value();
      ASSERT_EQ(obj_meta->size, expected_dets.size());

      for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];
        int catId = int(expected_dets[det_index]["category_id"]);

        cvtdl_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]),
            .y2 = float(bbox[3]),
            .score = float(expected_dets[det_index]["score"]),
        };

        auto comp = [=](cvtdl_object_info_t &info, cvtdl_bbox_t &bbox) {
          if (info.classes == catId && iou(info.bbox, bbox) >= bbox_threshold &&
              abs(info.bbox.score - bbox.score) <= score_threshold) {
            return true;
          }
          return false;
        };

        bool matched = match_dets(*obj_meta, expected_bbox, comp);
        EXPECT_TRUE(matched);
      }
      CVI_TDL_FreeCpp(obj_meta);
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl
