#include <dirent.h>
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

class FLTestSuite : public CVI_TDLModelTestSuite {
 public:
  typedef CVI_S32 (*InferenceFunc)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, cvtdl_face_t *);
  struct ModelInfo {
    InferenceFunc inference;
    CVI_TDL_SUPPORTED_MODEL_E index;
    std::string model_path;
  };

  FLTestSuite() : CVI_TDLModelTestSuite("reg_daily_fl.json", "reg_daily_fl") {}

  virtual ~FLTestSuite() = default;

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

  float sqrt3(const float x) {
    union {
      int i;
      float x;
    } u;

    u.x = x;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    return u.x;
  }

  float compute_nme(cvtdl_pts_t *labels, cvtdl_pts_t *pts) {
    float sum = 0.0;

    for (int i = 0; i < 5; ++i) {
      float _dist = 0.0;
      _dist += (float)((labels->x[i] - pts->x[i]) * (labels->x[i] - pts->x[i]));
      _dist += (float)((labels->y[i] - pts->y[i]) * (labels->y[i] - pts->y[i]));
      sum += sqrt3(_dist);
    }
    float _nme = sum / 5;
    float dist = sqrt3((float)((labels->x[0] - labels->x[1]) * (labels->x[0] - labels->x[1]) +
                               (labels->y[0] - labels->y[1]) * (labels->y[0] - labels->y[1])));
    _nme /= dist;
    return _nme;
  }
};

FLTestSuite::ModelInfo FLTestSuite::getModel(const std::string &model_name) {
  ModelInfo model_info;
  std::string model_path = (m_model_dir / model_name).string();
  model_info.index = CVI_TDL_SUPPORTED_MODEL_FACELANDMARKER;
  model_info.inference = CVI_TDL_FaceLandmarker;
  model_info.model_path = model_path;
  return model_info;
}

TEST_F(FLTestSuite, open_close_model) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name =
        std::string(std::string(test_config["reg_config"][0]["model_name"][0]).c_str());
    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_TDL_SUPPORTED_MODEL_END);

    TDLModelHandler tdlmodel(m_tdl_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    const char *model_path_get = CVI_TDL_GetModelPath(m_tdl_handle, model_info.index);

    EXPECT_PRED2([](auto s1, auto s2) { return s1 == s2; }, model_info.model_path,
                 std::string(model_path_get));
  }
}

TEST_F(FLTestSuite, inference_and_accuracy) {
  for (size_t test_idx = 0; test_idx < m_json_object.size(); test_idx++) {
    auto test_config = m_json_object[test_idx];
    std::string model_name =
        std::string(std::string(test_config["reg_config"][0]["model_name"][0]).c_str());

    ModelInfo model_info = getModel(model_name);
    ASSERT_LT(model_info.index, CVI_TDL_SUPPORTED_MODEL_END);

    TDLModelHandler tdlmodel(m_tdl_handle, model_info.index, model_info.model_path.c_str(), false);
    ASSERT_NO_FATAL_FAILURE(tdlmodel.open());

    float nme_threshold = float(test_config["reg_config"][0]["nme_threshold"]);
    float nme = 0.0;
    for (size_t img_idx = 0; img_idx < test_config["reg_config"][0]["test_images"].size();
         img_idx++) {
      std::string image_path =
          (m_image_dir / std::string(test_config["reg_config"][0]["test_images"][img_idx]))
              .string();

      Image image_rgb(image_path, PIXEL_FORMAT_RGB_888);
      ASSERT_TRUE(image_rgb.open());
      TDLObject<cvtdl_face_t> face_meta;
      init_face_meta(face_meta, 1);
      face_meta->width = image_rgb.getFrame()->stVFrame.u32Width;
      face_meta->height = image_rgb.getFrame()->stVFrame.u32Height;
      face_meta->info[0].bbox.x1 = (double)test_config["reg_config"][0]["bbox"][img_idx][0];
      face_meta->info[0].bbox.x2 = (double)test_config["reg_config"][0]["bbox"][img_idx][1];
      face_meta->info[0].bbox.y1 = (double)test_config["reg_config"][0]["bbox"][img_idx][2];
      face_meta->info[0].bbox.y2 = (double)test_config["reg_config"][0]["bbox"][img_idx][3];
      face_meta->info[0].pts.size = 5;
      face_meta->info[0].pts.x[0] = (float)test_config["reg_config"][0]["face_pts"][img_idx][0];
      face_meta->info[0].pts.x[1] = (float)test_config["reg_config"][0]["face_pts"][img_idx][1];
      face_meta->info[0].pts.x[2] = (float)test_config["reg_config"][0]["face_pts"][img_idx][2];
      face_meta->info[0].pts.x[3] = (float)test_config["reg_config"][0]["face_pts"][img_idx][3];
      face_meta->info[0].pts.x[4] = (float)test_config["reg_config"][0]["face_pts"][img_idx][4];
      face_meta->info[0].pts.y[0] = (float)test_config["reg_config"][0]["face_pts"][img_idx][5];
      face_meta->info[0].pts.y[1] = (float)test_config["reg_config"][0]["face_pts"][img_idx][6];
      face_meta->info[0].pts.y[2] = (float)test_config["reg_config"][0]["face_pts"][img_idx][7];
      face_meta->info[0].pts.y[3] = (float)test_config["reg_config"][0]["face_pts"][img_idx][8];
      face_meta->info[0].pts.y[4] = (float)test_config["reg_config"][0]["face_pts"][img_idx][9];
      ASSERT_EQ(model_info.inference(m_tdl_handle, image_rgb.getFrame(), face_meta),
                CVI_TDL_SUCCESS);

      nme += compute_nme(&(face_meta->info[0].pts), &(face_meta->dms->landmarks_5));
      CVI_TDL_FreeCpp(face_meta);
    }
    EXPECT_EQ(nme < nme_threshold, true);
  }
}
}  // namespace unitest
}  // namespace cvitdl
