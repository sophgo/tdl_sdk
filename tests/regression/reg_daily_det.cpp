#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "utils/tdl_log.hpp"

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
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();

    det_ = TDLModelFactory::getInstance().getModel(
        model_id, model_path);  // One model id may correspond to multiple
                                // models with different sizes
    ASSERT_NE(det_, nullptr);
    det_->setModelThreshold(m_json_object["model_score_threshold"]);
  }

  nlohmann::ordered_json convertDetectionResult(
      const std::shared_ptr<ModelOutputInfo> &out_data) {
    nlohmann::ordered_json result;  // is a list,contains bbox,conf,class_id
    if (out_data->getType() == ModelOutputType::OBJECT_DETECTION) {
      std::shared_ptr<ModelBoxInfo> obj_meta =
          std::static_pointer_cast<ModelBoxInfo>(out_data);
      LOGI("obj_meta->bboxes.size: %d", obj_meta->bboxes.size());
      for (const auto &box : obj_meta->bboxes) {
        nlohmann::ordered_json item;
        item["bbox"] = {box.x1, box.y1, box.x2, box.y2};
        item["score"] = box.score;
        item["class_id"] = box.class_id;
        LOGI("bbox: %f %f %f %f, score: %f, class_id: %d", box.x1, box.y1,
             box.x2, box.y2, box.score, box.class_id);
        result.push_back(item);
      }
    } else if (out_data->getType() ==
               ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS) {
      std::shared_ptr<ModelBoxLandmarkInfo> obj_meta =
          std::static_pointer_cast<ModelBoxLandmarkInfo>(out_data);
      for (const auto &box : obj_meta->box_landmarks) {
        nlohmann::ordered_json item;
        item["bbox"] = {box.x1, box.y1, box.x2, box.y2};
        item["score"] = box.score;
        item["class_id"] = box.class_id;
        LOGI("bbox: %f %f %f %f, conf: %f, class_id: %d", box.x1, box.y1,
             box.x2, box.y2, box.score, box.class_id);
        result.push_back(item);
      }
    } else {
      std::cout << "Unsupported output type: "
                << static_cast<int>(out_data->getType()) << std::endl;
      return nlohmann::ordered_json();
    }
    return result;
  }

  virtual void TearDown() {}
};

TEST_F(DetectionTestSuite, accuracy) {
  const float reg_nms_threshold = m_json_object["reg_nms_threshold"];
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  TestFlag test_flag = CVI_TDLTestContext::getInstance().getTestFlag();
  nlohmann::ordered_json results;
  LOGIP("test_flag: %d", static_cast<int>(test_flag));
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    LOGIP("checkToGetProcessResult failed");
    return;
  }
  size_t sample_num = results.size();
  LOGIP("regression sample num: %zu", sample_num);
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    auto expected_dets = iter.value();

    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();

    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);

    LOGIP("image_path: %s\n", image_path.c_str());

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    EXPECT_EQ(det_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::OBJECT_DETECTION ||
                out_type == ModelOutputType::OBJECT_DETECTION_WITH_LANDMARKS);
    LOGI("out_type: %d", static_cast<int>(out_type));
    if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
      LOGI("generate function res,image_path: %s\n", image_path.c_str());
      nlohmann::ordered_json result = convertDetectionResult(out_data[0]);
      iter.value() = result;
      continue;
    }

    std::vector<std::vector<float>> gt_dets;
    std::vector<std::vector<float>> pred_dets;

    // std::cout << "expected_dets: " << expected_dets << std::endl;
    if (out_type == ModelOutputType::OBJECT_DETECTION) {
      std::shared_ptr<ModelBoxInfo> obj_meta =
          std::static_pointer_cast<ModelBoxInfo>(out_data[0]);
      LOGI("obj_meta->bboxes.size: %d", obj_meta->bboxes.size());

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

      for (const auto &box_landmark : obj_meta->box_landmarks) {
        pred_dets.push_back({box_landmark.x1, box_landmark.y1, box_landmark.x2,
                             box_landmark.y2, box_landmark.score, class_id});
      }
    } else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
      return;
    }

    for (const auto &det : expected_dets) {
      // 检查字段是否存在
      if (!det.contains("bbox") || !det.contains("score")) {
        LOGE("Missing required fields in landmark detection data");
        continue;
      }
      if (det["bbox"].size() != 4) {
        LOGE("Invalid bbox size: %zu", det["bbox"].size());
        continue;
      }

      float bbox_x1 = det["bbox"][0];
      float bbox_y1 = det["bbox"][1];
      float bbox_x2 = det["bbox"][2];
      float bbox_y2 = det["bbox"][3];
      float score = det["score"];
      float class_id = 0;
      if (det.contains("class_id")) {
        class_id = det["class_id"];
      }
      gt_dets.push_back({bbox_x1, bbox_y1, bbox_x2, bbox_y2, score, class_id});
    }
    // EXPECT_TRUE(matchObjects(gt_dets, pred_dets, reg_nms_threshold,
    //                          reg_score_diff_threshold));

    break;
  }
  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}

}  // namespace unitest
}  // namespace cvitdl
