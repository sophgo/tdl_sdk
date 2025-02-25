#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/cvi_tdl_types_mem.h"
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
      : CVI_TDLModelTestSuite("reg_daily_landmarkdet2.json", 
                              "reg_daily_fl") {}

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

      cvtdl_face_t face_pre;
      face_pre.info = (cvtdl_face_info_t*)malloc(sizeof(cvtdl_face_info_t));
      face_pre.info[0].pts.x = (float*)malloc(sizeof(float) * 5);
      face_pre.info[0].pts.y = (float*)malloc(sizeof(float) * 5);
      auto expected_dets = iter.value();
      const auto &gt_det = expected_dets[0];
      face_pre.width = (int)frame->getWidth();
      face_pre.height = (int)frame->getHeight();
      face_pre.info[0].bbox.x1 = (double)gt_det["bbox"][0];
      face_pre.info[0].bbox.x2 = (double)gt_det["bbox"][1];
      face_pre.info[0].bbox.y1 = (double)gt_det["bbox"][2];
      face_pre.info[0].bbox.y2 = (double)gt_det["bbox"][3];
      face_pre.info[0].pts.size = 5;
      face_pre.info[0].pts.x[0] = (float)gt_det["face_pts"][0];
      face_pre.info[0].pts.x[1] = (float)gt_det["face_pts"][1];
      face_pre.info[0].pts.x[2] = (float)gt_det["face_pts"][2];
      face_pre.info[0].pts.x[3] = (float)gt_det["face_pts"][3];
      face_pre.info[0].pts.x[4] = (float)gt_det["face_pts"][4];
      face_pre.info[0].pts.y[0] = (float)gt_det["face_pts"][5];
      face_pre.info[0].pts.y[1] = (float)gt_det["face_pts"][6];
      face_pre.info[0].pts.y[2] = (float)gt_det["face_pts"][7];
      face_pre.info[0].pts.y[3] = (float)gt_det["face_pts"][8];
      face_pre.info[0].pts.y[4] = (float)gt_det["face_pts"][9];
      
      int x1 = face_pre.info[0].bbox.x1;
      int x2 = face_pre.info[0].bbox.x2;
      int y1 = face_pre.info[0].bbox.y1;
      int y2 = face_pre.info[0].bbox.y2;

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
      std::vector<void *> out_data;
      std::vector<std::shared_ptr<BaseImage>> input_images;
      input_images.push_back(face_crop);

      EXPECT_EQ(m_model->inference(input_images, out_data), 0);
      EXPECT_EQ(out_data.size(), 1);

      cvtdl_face_info_t *face_info = (cvtdl_face_info_t *)out_data[0];
      // 将点映射回原图
      for(int i = 0 ; i < 5; i++) {
        face_info->pts.x[i] = face_info->pts.x[i] + crop_x1;
        face_info->pts.y[i] = face_info->pts.y[i] + crop_y1;
        // std::cout << "face_meta_pts:" << face_info->pts.x[i] << " "<< face_info->pts.y[i]<< std::endl;
      }
      
      nme += compute_nme(&(face_pre.info[0].pts), &(face_info->pts));

      model_factory.releaseOutput(
          TDL_MODEL_TYPE_FACE_LANDMARKER_LANDMARKERDETV2, out_data);
      free(face_pre.info[0].pts.x);
      free(face_pre.info[0].pts.y);
      face_pre.info[0].pts.x = NULL;
      face_pre.info[0].pts.y = NULL;
      free(face_pre.info);
      face_pre.info = NULL;
      EXPECT_EQ(nme < nme_threshold, true);
    } 
  }
}

}  // namespace unitest
}  // namespace cvitdl
