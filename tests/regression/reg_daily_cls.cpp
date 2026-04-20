#include <gtest/gtest.h>
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
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class ClassificationTestSuite : public CVI_TDLModelTestSuite {
 public:
  ClassificationTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~ClassificationTestSuite() = default;

  std::string m_model_path;
  std::shared_ptr<BaseModel> cls_;

  std::string model_id_str;
  ModelType model_id_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    model_id_str = std::string(m_json_object["model_id"]);
    model_id_ = modelTypeFromString(model_id_str);
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();
    cls_ = TDLModelFactory::getInstance().getModel(
        model_id_, model_path);  // One model id may correspond to multiple
                                 // models with different sizes

    ASSERT_NE(cls_, nullptr);
  }

  void TearDown() override {}

  nlohmann::ordered_json convertClassificationResult(
      const std::shared_ptr<ModelClassificationInfo>& out_data) {
    nlohmann::ordered_json result;
    auto obj_meta = std::static_pointer_cast<ModelClassificationInfo>(out_data);
    nlohmann::ordered_json item;
    item["score"] = obj_meta->topk_scores[0];
    item["class_id"] = obj_meta->topk_class_ids[0];
    result.push_back(item);
    return result;
  }
};

TEST_F(ClassificationTestSuite, accuracy) {
  std::string image_dir = (m_image_dir / m_json_object["image_dir"])
                              .string();  // 使用统一的image_dir
  std::string platform = get_platform_str();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();

  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }

  nlohmann::ordered_json output_results;

  for (auto iter = results.begin(); iter != results.end(); ++iter) {
    std::string path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();

    auto data = loadInputData(path);
    ASSERT_NE(data, nullptr);

    if (model_id_str == "VAD_FSMN") {
      // VAD-specific inference and evaluation
      std::shared_ptr<ModelOutputInfo> output_info;
      ASSERT_EQ(cls_->inference(data, output_info), 0);

      auto vad_meta = std::static_pointer_cast<ModelVADInfo>(output_info);

      if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
        nlohmann::ordered_json item;
        item["class_id"] = vad_meta->has_segments ? 1 : 0;
        output_results[iter.key()] = item;
      } else {
        int expected =
            iter.value().is_number_integer()
                ? iter.value().get<int>()
                : (iter.value().is_object() && iter.value().contains("class_id")
                       ? iter.value()["class_id"].get<int>()
                       : -1);
        ASSERT_EQ(expected, vad_meta->has_segments ? 1 : 0);
      }
    } else {
      // Classification-specific inference and evaluation
      std::vector<std::shared_ptr<BaseImage>> input_images = {
          std::static_pointer_cast<BaseImage>(data)};
      std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
      EXPECT_EQ(cls_->inference(input_images, out_data), 0);
      EXPECT_EQ(out_data.size(), 1u);
      EXPECT_EQ(out_data[0]->getType(), ModelOutputType::CLASSIFICATION);

      auto cls_meta =
          std::static_pointer_cast<ModelClassificationInfo>(out_data[0]);

      if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
        output_results[iter.key()] = convertClassificationResult(cls_meta);
      } else {
        auto expected_cls = iter.value()[0];
        int gt_class_id = expected_cls["class_id"];
        float gt_score = expected_cls["score"];

        int pred_class_id = cls_meta->topk_class_ids[0];
        float pred_score = cls_meta->topk_scores[0];

        EXPECT_EQ(gt_class_id, pred_class_id);
        EXPECT_NEAR(gt_score, pred_score, 0.1f);
      }
    }
  }

  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = output_results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  } else {
    LOGIP("Processed items: %d", results.size());
  }
}

TEST_F(ClassificationTestSuite, performance) {
  std::string model_path = m_model_dir.string() + "/" + gen_model_dir() + "/" +
                           m_json_object["model_name"].get<std::string>() +
                           gen_model_suffix();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    LOGIP("checkToGetProcessResult failed");
    return;
  }

  auto it = results.begin();
  std::string image_path =
      (m_image_dir / m_json_object["image_dir"] / it.key()).string();
  LOGIP("image path : %s", image_path.c_str());
  std::shared_ptr<BaseImage> frame = loadInputData(image_path);
  if (!frame) {
    LOGE("performance: failed to read image %s", image_path.c_str());
  }

  if (model_id_str == "CLS_SOUND_COMMAND_NIHAOSHIYUN" ||
      model_id_str == "CLS_SOUND_COMMAND_XIAOAIXIAOAI") {
    run_performance(model_path, frame, cls_, 2000.0);
  } else if (model_id_str == "CLS_SOUND_BABAY_CRY") {
    run_performance(model_path, frame, cls_, 3000.0);
  } else if (model_id_str == "VAD_FSMN") {
    run_performance(model_path, frame, cls_, 1000.0);
  } else {
    run_performance(model_path, frame, cls_);
  }
}

}  // namespace unitest
}  // namespace cvitdl