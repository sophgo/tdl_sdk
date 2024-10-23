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

class ScrfdDetTestSuite : public CVI_TDLModelTestSuite {
 public:
  ScrfdDetTestSuite() : CVI_TDLModelTestSuite("reg_daily_scrfdface.json", "reg_daily_scrfdface") {}

  virtual ~ScrfdDetTestSuite() = default;

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

TEST_F(ScrfdDetTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    ASSERT_EQ(
        CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, m_model_path.c_str()),
        CVI_TDL_SUCCESS)
        << "failed to set model path: " << m_model_path.c_str();
    ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE), CVI_TDL_SUCCESS);
  }
}

TEST_F(ScrfdDetTestSuite, inference) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    TDLModelHandler tdlmodel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, m_model_path.c_str(),
                             false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    auto results = m_json_object[test_index]["results"];

    std::string image_path = (m_image_dir / results.begin().key()).string();

    {
      Image frame(image_path, PIXEL_FORMAT_RGB_888_PLANAR);
      ASSERT_TRUE(frame.open());
      cvtdl_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvtdl_face_t));
      EXPECT_EQ(CVI_TDL_FaceDetection(m_tdl_handle, frame.getFrame(),
                                      CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, &face_meta),
                CVI_TDL_SUCCESS);
    }

    {
      Image frame(image_path, PIXEL_FORMAT_BGR_888);
      ASSERT_TRUE(frame.open());
      cvtdl_face_t face_meta;
      memset(&face_meta, 0, sizeof(cvtdl_face_t));
      EXPECT_EQ(CVI_TDL_FaceDetection(m_tdl_handle, frame.getFrame(),
                                      CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, &face_meta),
                CVI_TDL_SUCCESS);
    }
  }
}

TEST_F(ScrfdDetTestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name = std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    ASSERT_EQ(
        CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, m_model_path.c_str()),
        CVI_TDL_SUCCESS);
    const float bbox_threshold = 0.90;
    const float score_threshold = 0.1;
    auto results = m_json_object[test_index]["results"];

    for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();

      Image frame(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(frame.open());

      TDLObject<cvtdl_face_t> face_meta;

      ASSERT_EQ(CVI_TDL_FaceDetection(m_tdl_handle, frame.getFrame(),
                                      CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, face_meta),
                CVI_TDL_SUCCESS);

      auto expected_dets = iter.value();
      ASSERT_EQ(face_meta->size, expected_dets.size());

      for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];

        cvtdl_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]),
            .y2 = float(bbox[3]),
            .score = float(expected_dets[det_index]["score"]),
        };

        auto comp = [=](cvtdl_face_info_t &info, cvtdl_bbox_t &bbox) {
          if (iou(info.bbox, bbox) >= bbox_threshold &&
              abs(info.bbox.score - bbox.score) <= score_threshold) {
            return true;
          }
          return false;
        };
        EXPECT_TRUE(match_dets(*face_meta, expected_bbox, comp))
            << "Error!"
            << "\n"
            << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1 << ", "
            << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n"
            << "score: " << expected_bbox.score << "\n"
            << "[" << face_meta->info[det_index].bbox.x1 << ","
            << face_meta->info[det_index].bbox.y1 << "," << face_meta->info[det_index].bbox.x2
            << "," << face_meta->info[det_index].bbox.y2 << ","
            << face_meta->info[det_index].bbox.score << "],\n";
      }
      CVI_TDL_FreeCpp(face_meta);
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl
