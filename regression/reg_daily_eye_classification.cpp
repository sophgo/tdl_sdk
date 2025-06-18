#include <fstream>
#include <string>
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

class EyeCTestSuite : public CVI_TDLModelTestSuite {
 public:
  typedef CVI_S32 (*InferenceFunc)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, cvtdl_face_t *);
  struct ModelInfo {
    InferenceFunc inference;
    CVI_TDL_SUPPORTED_MODEL_E index;
    std::string model_path;
  };

  EyeCTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_eye_classification.json", "reg_daily_eye_classification") {
  }

  virtual ~EyeCTestSuite() = default;

 protected:
  virtual void SetUp() {
    m_tdl_handle = NULL;
    ASSERT_EQ(CVI_TDL_CreateHandle2(&m_tdl_handle, 1, 0), CVI_TDL_SUCCESS);
  }

  virtual void TearDown() {
    CVI_TDL_DestroyHandle(m_tdl_handle);
    m_tdl_handle = NULL;
  }

  ModelInfo getModel(const std::string &model_name);
};

EyeCTestSuite::ModelInfo EyeCTestSuite::getModel(const std::string &model_name) {
  ModelInfo model_info;
  std::string model_path = (m_model_dir / model_name).string();
  model_info.index = CVI_TDL_SUPPORTED_MODEL_EYECLASSIFICATION;
  model_info.inference = CVI_TDL_EyeClassification;
  model_info.model_path = model_path;
  return model_info;
}

TEST_F(EyeCTestSuite, open_close_model) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name =
        std::string(std::string(test_config["reg_config"][0]["model_name"][0]).c_str());
    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_TDL_SUPPORTED_MODEL_END);

    TDLModelHandler tdlmodel(m_tdl_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    const char *model_path_get = CVI_TDL_GetModelPath(m_tdl_handle, model_info.index);

    EXPECT_PRED2([](const std::string &s1, const std::string &s2) { return s1 == s2; },
                 model_info.model_path, std::string(model_path_get));
  }
}

TEST_F(EyeCTestSuite, inference_and_accuracy) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name =
        std::string(std::string(test_config["reg_config"][0]["model_name"][0]).c_str());

    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_TDL_SUPPORTED_MODEL_END);

    TDLModelHandler tdlmodel(m_tdl_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    float threshold = float(test_config["reg_config"][0]["threshold"]);
    for (size_t img_idx = 0; img_idx < test_config["reg_config"][0]["test_images"].size();
         img_idx++) {
      std::string image_path =
          (m_image_dir / std::string(test_config["reg_config"][0]["test_images"][img_idx]))
              .string();

      int expected_res = int(test_config["reg_config"][0]["expected_results"][img_idx]);
      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image_rgb.open());

      TDLObject<cvtdl_face_t> face_meta;
      init_face_meta(face_meta, 1);

      face_meta->width = 1280;
      face_meta->height = 720;
      face_meta->info[0].bbox.x1 = (double)test_config["reg_config"][0]["bbox"][img_idx][0];
      face_meta->info[0].bbox.x2 = (double)test_config["reg_config"][0]["bbox"][img_idx][1];
      face_meta->info[0].bbox.y1 = (double)test_config["reg_config"][0]["bbox"][img_idx][2];
      face_meta->info[0].bbox.y2 = (double)test_config["reg_config"][0]["bbox"][img_idx][3];
      face_meta->dms->landmarks_5.x[0] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][0];
      face_meta->dms->landmarks_5.x[1] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][1];
      face_meta->dms->landmarks_5.x[2] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][2];
      face_meta->dms->landmarks_5.x[3] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][3];
      face_meta->dms->landmarks_5.x[4] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][4];
      face_meta->dms->landmarks_5.y[0] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][5];
      face_meta->dms->landmarks_5.y[1] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][6];
      face_meta->dms->landmarks_5.y[2] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][7];
      face_meta->dms->landmarks_5.y[3] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][8];
      face_meta->dms->landmarks_5.y[4] =
          (float)test_config["reg_config"][0]["face_landmarks"][img_idx][9];

      ASSERT_EQ(model_info.inference(m_tdl_handle, image_rgb.getFrame(), face_meta),
                CVI_TDL_SUCCESS);
      EXPECT_EQ(
          ((face_meta->dms->leye_score > threshold) || (face_meta->dms->reye_score > threshold)),
          expected_res);
      CVI_TDL_FreeDMS(face_meta->dms);
      CVI_TDL_FreeCpp(face_meta);
    }
  }
}
}  // namespace unitest
}  // namespace cvitdl
