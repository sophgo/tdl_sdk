#include <gtest.h>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_model_id.hpp"
#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_log.hpp"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class KeypointTestSuite : public CVI_TDLModelTestSuite {
 public:
  KeypointTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~KeypointTestSuite() = default;

  std::string m_model_path;
  std::shared_ptr<BaseModel> keypoint_;
  TDLModelFactory model_factory_;
  ModelType model_id_;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model_name"]);
    std::string model_path =
        (m_model_dir / fs::path(model_name + gen_model_suffix())).string();
    std::string model_id_name = std::string(m_json_object["model_id"]);
    model_id_ = stringToModelType(model_id_name);
    keypoint_ = model_factory_.getModel(model_id_, model_path);
    ASSERT_NE(keypoint_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(KeypointTestSuite, accuracy) {
  const float score_threshold = m_json_object["score_threshold"];
  const float position_threshold = m_json_object["position_threshold"];
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto results = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    printf("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame = getInputData(image_path, model_id_);
    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    auto expected_keypoint = iter.value();
    std::vector<float> gt_keypoints_x;
    std::vector<float> gt_keypoints_y;
    std::vector<float> gt_keypoints_score;
    gt_keypoints_x = expected_keypoint["keypoints_x"].get<std::vector<float>>();
    gt_keypoints_y = expected_keypoint["keypoints_y"].get<std::vector<float>>();
    gt_keypoints_score =
        expected_keypoint["keypoints_score"].get<std::vector<float>>();

    std::vector<float> pred_keypoints_x;
    std::vector<float> pred_keypoints_y;
    std::vector<float> pred_keypoints_score;

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    EXPECT_EQ(keypoint_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::OBJECT_LANDMARKS ||
                out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS);

    if (out_type == ModelOutputType::OBJECT_LANDMARKS) {
      std::shared_ptr<ModelLandmarksInfo> keypoint_meta =
          std::static_pointer_cast<ModelLandmarksInfo>(out_data[0]);
      pred_keypoints_x = keypoint_meta->landmarks_x;
      pred_keypoints_y = keypoint_meta->landmarks_y;
      pred_keypoints_score = keypoint_meta->landmarks_score;
    } else if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data[0]);
      ObjectBoxLandmarkInfo keypoint_meta = obj_meta->box_landmarks[0];
      pred_keypoints_x = keypoint_meta.landmarks_x;
      pred_keypoints_y = keypoint_meta.landmarks_y;
      pred_keypoints_score = keypoint_meta.landmarks_score;
    } else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
      return;
    }

    float max_x = 0.0f, max_y = 0.0f;
    for (const auto& x : pred_keypoints_x) {
      max_x = std::max(max_x, x);
    }
    for (const auto& y : pred_keypoints_y) {
      max_y = std::max(max_y, y);
    }

    if (max_x > 1.0f || max_y > 1.0f) {
      float img_width = frame->getWidth();
      float img_height = frame->getHeight();

      for (auto& x : pred_keypoints_x) {
        x /= img_width;
      }
      for (auto& y : pred_keypoints_y) {
        y /= img_height;
      }
    }

    EXPECT_TRUE(matchKeypoints(
        gt_keypoints_x, gt_keypoints_y, gt_keypoints_score, pred_keypoints_x,
        pred_keypoints_y, pred_keypoints_score, position_threshold, score_threshold));
  }
}

}  // namespace unitest
}  // namespace cvitdl
