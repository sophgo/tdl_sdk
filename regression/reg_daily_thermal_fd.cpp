#include <experimental/filesystem>
#include <fstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "gtest.h"
#include "json.hpp"
#include "raii.hpp"

#define MATCH_IOU_THRESHOLD 0.85
#define MATCH_SCORE_BIAS 0.02

static float iou(cvtdl_bbox_t &bbox1, cvtdl_bbox_t &bbox2) {
  float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
  float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
  float inter_x1 = MAX2(bbox1.x1, bbox2.x1);
  float inter_y1 = MAX2(bbox1.y1, bbox2.y1);
  float inter_x2 = MIN2(bbox1.x2, bbox2.x2);
  float inter_y2 = MIN2(bbox1.y2, bbox2.y2);
  float area_inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
#if 0
  printf(" area1: %f,  area2: %f,  area_inter: %f,  iou: %f\n", 
    area1, area2, area_inter,
    area_inter / (area1 + area2 - area_inter));
#endif
  return area_inter / (area1 + area2 - area_inter);
}

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class ThermalFaceDetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  ThermalFaceDetectionTestSuite()
      : CVI_TDLModelTestSuite("daily_reg_ThermalFD.json", "reg_daily_thermal_fd") {}

  virtual ~ThermalFaceDetectionTestSuite() = default;

  std::string m_model_path;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["reg_config"][0]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
    ASSERT_EQ(CVI_TDL_SetVpssTimeout(m_tdl_handle, 1000), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }
};

TEST_F(ThermalFaceDetectionTestSuite, open_close_model) {
  ASSERT_EQ(
      CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, m_model_path.c_str()),
      CVI_TDL_SUCCESS)
      << "failed to set model path: " << m_model_path;

  const char *model_path_get =
      CVI_TDL_GetModelPath(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE);

  EXPECT_PRED2([](const std::string &s1, const std::string &s2) { return s1 == s2; }, m_model_path,
               std::string(model_path_get));

  ASSERT_EQ(CVI_TDL_CloseModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE), CVI_TDL_SUCCESS);
}

TEST_F(ThermalFaceDetectionTestSuite, accruacy) {
  ASSERT_EQ(
      CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, m_model_path.c_str()),
      CVI_TDL_SUCCESS);
  ASSERT_EQ(CVI_TDL_SetSkipVpssPreprocess(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, false),
            CVI_TDL_SUCCESS);
  ASSERT_EQ(CVI_TDL_SetModelThreshold(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, 0.5),
            CVI_TDL_SUCCESS);

  int img_num = int(m_json_object["reg_config"][0]["image_num"]);
  for (int img_idx = 0; img_idx < img_num; img_idx++) {
    std::string image_path = std::string(m_json_object["reg_config"][0]["test_images"][img_idx]);
    image_path = (m_image_dir / image_path).string();
    // printf("[%d] %s\n", img_idx, image_path.c_str());

    Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
    ASSERT_TRUE(image_rgb.open());
    VIDEO_FRAME_INFO_S *vframe = image_rgb.getFrame();

    TDLObject<cvtdl_face_t> face_meta;
    ASSERT_EQ(
        CVI_TDL_FaceDetection(m_tdl_handle, vframe, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, face_meta),
        CVI_TDL_SUCCESS);

    uint32_t expected_bbox_num =
        uint32_t(m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_num"]);
    ASSERT_EQ(expected_bbox_num, face_meta->size);

    cvtdl_bbox_t *expected_result = new cvtdl_bbox_t[expected_bbox_num];
    for (uint32_t i = 0; i < expected_bbox_num; i++) {
      expected_result[i].score = float(
          m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["score"]);
      expected_result[i].x1 =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["x1"]);
      expected_result[i].y1 =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["y1"]);
      expected_result[i].x2 =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["x2"]);
      expected_result[i].y2 =
          float(m_json_object["reg_config"][0]["expected_results"][img_idx]["bbox_info"][i]["y2"]);
#if 0
      printf("(%u) %f (%f,%f,%f,%f)\n", i, expected_result[i].score, 
        expected_result[i].x1, expected_result[i].y1, 
        expected_result[i].x2, expected_result[i].y2);
#endif
    }
    bool *match_result = new bool[expected_bbox_num];
    std::fill_n(match_result, expected_bbox_num, false);
    for (uint32_t j = 0; j < face_meta->size; j++) {
      bool is_match = false;
      for (uint32_t i = 0; i < expected_bbox_num; i++) {
        if (match_result[i]) continue;
        if (iou(face_meta->info[j].bbox, expected_result[i]) < MATCH_IOU_THRESHOLD) continue;
        if (ABS(face_meta->info[j].bbox.score - expected_result[i].score) < MATCH_SCORE_BIAS) {
          match_result[i] = true;
          is_match = true;
        }
        break;
      }
      if (!is_match) break;
    }
    for (uint32_t i = 0; i < expected_bbox_num; i++) {
      ASSERT_EQ(match_result[i], true);
    }

    CVI_TDL_FreeCpp(face_meta);
    delete[] expected_result;
    delete[] match_result;
  }
}

}  // namespace unitest
}  // namespace cvitdl
