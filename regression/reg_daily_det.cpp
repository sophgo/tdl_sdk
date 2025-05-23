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

class DetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  DetectionTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~DetectionTestSuite() = default;

  std::shared_ptr<BaseModel> det_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id = std::string(m_json_object["model_id"]);
    det_ = TDLModelFactory::getInstance().getModel(model_id);
    ASSERT_NE(det_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(DetectionTestSuite, accuracy) {
  const float bbox_threshold = m_json_object["bbox_threshold"];
  const float score_threshold = m_json_object["score_threshold"];

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto results = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    LOGIP("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    EXPECT_EQ(det_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::OBJECT_DETECTION ||
                out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS);
    std::vector<std::vector<float>> gt_dets;
    std::vector<std::vector<float>> pred_dets;
    if (out_type == ModelOutputType::OBJECT_DETECTION) {
      std::shared_ptr<ModelBoxInfo> obj_meta =
          std::static_pointer_cast<ModelBoxInfo>(out_data[0]);
      auto expected_dets = iter.value();
      for (const auto &det : expected_dets) {
        gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                           det["bbox"][3], det["score"], det["class_id"]});
      }
      for (uint32_t det_index = 0; det_index < obj_meta->bboxes.size();
           det_index++) {
        pred_dets.push_back(
            {obj_meta->bboxes[det_index].x1, obj_meta->bboxes[det_index].y1,
             obj_meta->bboxes[det_index].x2, obj_meta->bboxes[det_index].y2,
             obj_meta->bboxes[det_index].score,
             float(obj_meta->bboxes[det_index].class_id)});
      }
    } else if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      float class_id = 0;
      std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data[0]);
      auto expected_dets = iter.value();
      for (const auto &det : expected_dets) {
        gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                           det["bbox"][3], det["score"], class_id});
      }
      for (const auto &box_landmark : obj_meta->box_landmarks) {
        pred_dets.push_back({box_landmark.x1, box_landmark.y1, box_landmark.x2,
                             box_landmark.y2, box_landmark.score, class_id});
      }
    } else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
      return;
    }
    EXPECT_TRUE(
        matchObjects(gt_dets, pred_dets, bbox_threshold, score_threshold));
  }
}

}  // namespace unitest
}  // namespace cvitdl
