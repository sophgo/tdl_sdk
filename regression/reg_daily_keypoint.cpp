#include <gtest.h>
#include <fstream>
#include <string>
#include <unordered_map>

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

 protected:
  virtual void SetUp() {
    float_precesion_num_ = 4;
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id_str = std::string(m_json_object["model_id"]);

    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();
    keypoint_ = TDLModelFactory::getInstance().getModel(
        model_id_str, model_path);  // One model id may correspond to multiple
                                    // models with different sizes

    ASSERT_NE(keypoint_, nullptr);
  }

  virtual void TearDown() {}

  nlohmann::ordered_json convertKeypointResult(
      const std::shared_ptr<ModelOutputInfo>& out_data, int img_width,
      int img_height) {
    nlohmann::ordered_json result;
    std::vector<float> keypoints_x;
    std::vector<float> keypoints_y;
    std::vector<float> keypoints_score;
    if (out_data->getType() == ModelOutputType::OBJECT_LANDMARKS) {
      std::shared_ptr<ModelLandmarksInfo> obj_meta =
          std::static_pointer_cast<ModelLandmarksInfo>(out_data);
      keypoints_x = obj_meta->landmarks_x;
      keypoints_y = obj_meta->landmarks_y;
      if (!obj_meta->attributes.empty()) {
        for (const auto& pair : obj_meta->attributes) {
          keypoints_score.push_back(pair.second);
        }
      } else {
        keypoints_score = obj_meta->landmarks_score;
      }
    } else if (out_data->getType() ==
               ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
      keypoints_x = obj_meta->box_landmarks[0].landmarks_x;
      keypoints_y = obj_meta->box_landmarks[0].landmarks_y;
      keypoints_score = obj_meta->box_landmarks[0].landmarks_score;
    } else {
      std::cout << "Unsupported output type: "
                << static_cast<int>(out_data->getType()) << std::endl;
      return nlohmann::ordered_json();
    }

    for (size_t i = 0; i < keypoints_x.size(); i++) {
      float kpt_xi = std::max(keypoints_x[i], 0.0f);
      if (kpt_xi > 1.0) {
        kpt_xi = kpt_xi / img_width;
      }
      result["keypoints_x"].push_back(kpt_xi);
    }
    for (size_t i = 0; i < keypoints_y.size(); i++) {
      float kpt_yi = std::max(keypoints_y[i], 0.0f);
      if (kpt_yi > 1.0) {
        kpt_yi = kpt_yi / img_height;
      }
      result["keypoints_y"].push_back(kpt_yi);
    }
    result["keypoints_score"] = keypoints_score;
    return result;
  }
};

TEST_F(KeypointTestSuite, accuracy) {
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];
  const float reg_position_diff_threshold =
      m_json_object["reg_position_diff_threshold"];
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = gen_platform();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }

  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    printf("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame = loadInputData(image_path);
    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    auto expected_keypoint = iter.value();

    std::vector<float> pred_keypoints_x;
    std::vector<float> pred_keypoints_y;
    std::vector<float> pred_keypoints_score;

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    EXPECT_EQ(keypoint_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::OBJECT_LANDMARKS ||
                out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS);

    int img_width = static_cast<int>(frame->getWidth());
    int img_height = static_cast<int>(frame->getHeight());
    LOGI("img_width: %d, img_height: %d,test_flag: %d,result_type: %d",
         img_width, img_height, context.getTestFlag(),
         static_cast<int>(out_type));
    if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
      nlohmann::ordered_json result =
          convertKeypointResult(out_data[0], img_width, img_height);
      iter.value() = result;
      continue;
    }
    // std::cout << "expected_keypoint: " << expected_keypoint << std::endl;
    std::vector<float> gt_keypoints_x;
    std::vector<float> gt_keypoints_y;
    std::vector<float> gt_keypoints_score;
    gt_keypoints_x = expected_keypoint["keypoints_x"].get<std::vector<float>>();
    gt_keypoints_y = expected_keypoint["keypoints_y"].get<std::vector<float>>();
    gt_keypoints_score =
        expected_keypoint["keypoints_score"].get<std::vector<float>>();

    if (out_type == ModelOutputType::OBJECT_LANDMARKS) {
      std::shared_ptr<ModelLandmarksInfo> keypoint_meta =
          std::static_pointer_cast<ModelLandmarksInfo>(out_data[0]);
      pred_keypoints_x = keypoint_meta->landmarks_x;
      pred_keypoints_y = keypoint_meta->landmarks_y;
      if (!keypoint_meta->attributes.empty()) {
        for (const auto& pair : keypoint_meta->attributes) {
          pred_keypoints_score.push_back(pair.second);
        }
      } else {
        pred_keypoints_score = keypoint_meta->landmarks_score;
      }
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
      for (auto& x : pred_keypoints_x) {
        x /= img_width;
      }
      for (auto& y : pred_keypoints_y) {
        y /= img_height;
      }
    }
    EXPECT_TRUE(
        matchKeypoints(gt_keypoints_x, gt_keypoints_y, gt_keypoints_score,
                       pred_keypoints_x, pred_keypoints_y, pred_keypoints_score,
                       reg_position_diff_threshold, reg_score_diff_threshold));
  }
  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}

}  // namespace unitest
}  // namespace cvitdl
