#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "models/tdl_model_factory.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class FaceLandmarkerDet2TestSuite : public CVI_TDLModelTestSuite {
 public:
  FaceLandmarkerDet2TestSuite()
      : CVI_TDLModelTestSuite("reg_daily_landmarkdet2.json", "reg_daily_fl") {}

  virtual ~FaceLandmarkerDet2TestSuite() = default;

  std::string m_model_path;
  TDLModelFactory model_factory;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}

  float sqrt3(const float x) {
    union {
      int i;
      float x;
    } u;

    u.x = x;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    return u.x;
  }

  float compute_nme(float *gtx, float *gty, float *ptsx, float *ptsy) {
    float sum = 0.0;

    for (int i = 0; i < 5; ++i) {
      float _dist = 0.0;
      _dist += (float)((gtx[i] - ptsx[i]) * (gtx[i] - ptsx[i]));
      _dist += (float)((gty[i] - ptsy[i]) * (gty[i] - ptsy[i]));
      sum += sqrt3(_dist);
    }
    float _nme = sum / 5;
    float dist = sqrt3((gtx[0] - gtx[1]) * (gtx[0] - gtx[1]) +
                       (gty[0] - gty[1]) * (gty[0] - gty[1]));
    _nme /= dist;
    return _nme;
  }
};

TEST_F(FaceLandmarkerDet2TestSuite, accuracy) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    auto results = m_json_object[test_index]["results"];
    const float nme_threshold = m_json_object[test_index]["nme_threshold"];
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    std::string model_path = (m_model_dir / fs::path(model_name)).string();
    std::shared_ptr<BaseModel> m_model = model_factory.getModel(
        TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2, model_path);
    std::shared_ptr<BasePreprocessor> preprocessor = m_model->getPreprocessor();
    for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
         iter++) {
      float nme = 0.0;
      std::string image_path = (m_image_dir / iter.key()).string();
      std::cout << "image_path: " << image_path << std::endl;
      std::shared_ptr<BaseImage> frame =
          ImageFactory::readImage(image_path, true);

      ASSERT_NE(frame, nullptr);

      const auto &gt_det = iter.value();
      float x1 = (double)gt_det["bbox"][0];
      float x2 = (double)gt_det["bbox"][1];
      float y1 = (double)gt_det["bbox"][2];
      float y2 = (double)gt_det["bbox"][3];

      std::vector<float> gt_ptx;
      std::vector<float> gt_pty;
      for (int i = 0; i < 5; i++) {
        gt_ptx.push_back((float)gt_det["face_pts"][i]);
        gt_pty.push_back((float)gt_det["face_pts"][i + 5]);
      }

      int box_width = x2 - x1;
      int box_height = y2 - y1;
      int crop_size = int(std::max(box_width, box_height) * 1.2);
      int crop_x1 = x1 - (crop_size - box_width) / 2;
      int crop_y1 = y1 - (crop_size - box_height) / 2;
      int crop_x2 = x2 + (crop_size - box_width) / 2;
      int crop_y2 = y2 + (crop_size - box_height) / 2;
      crop_x1 = std::max(crop_x1, 0);
      crop_y1 = std::max(crop_y1, 0);
      crop_x2 = std::min(crop_x2, (int)frame->getWidth());
      crop_y2 = std::min(crop_y2, (int)frame->getHeight());

      std::shared_ptr<BaseImage> face_crop = preprocessor->crop(
          frame, crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1);

      // break;
      std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
      std::vector<std::shared_ptr<BaseImage>> input_images;
      input_images.push_back(face_crop);

      EXPECT_EQ(m_model->inference(input_images, out_data), 0);
      EXPECT_EQ(out_data.size(), 1);
      EXPECT_EQ(out_data[0]->getType(), ModelOutputType::OBJECT_LANDMARKS);
      std::shared_ptr<ModelLandmarksInfo> landmark_info =
          std::static_pointer_cast<ModelLandmarksInfo>(out_data[0]);
      std::vector<float> pred_ptx;
      std::vector<float> pred_pty;
      for (int i = 0; i < 5; i++) {
        pred_ptx.push_back(landmark_info->landmarks_x[i] + crop_x1);
        pred_pty.push_back(landmark_info->landmarks_y[i] + crop_y1);
      }

      nme += compute_nme(&gt_ptx[0], &gt_pty[0], &pred_ptx[0], &pred_pty[0]);
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl
