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

  ModelType model_id_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id_str = std::string(m_json_object["model_id"]);
    model_id_ = modelTypeFromString(model_id_str);
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();
    cls_ = TDLModelFactory::getInstance().getModel(
        model_id_, model_path);  // One model id may correspond to multiple
                                 // models with different sizes

    ASSERT_NE(cls_, nullptr);
  }

  virtual void TearDown() {}

  nlohmann::ordered_json convertClassificationResult(
      const std::shared_ptr<ModelClassificationInfo> &out_data) {
    nlohmann::ordered_json result;  // is a list,contains bbox,conf,class_id
    std::shared_ptr<ModelClassificationInfo> obj_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_data);
    nlohmann::ordered_json item;
    item["score"] = obj_meta->topk_scores[0];
    item["class_id"] = obj_meta->topk_class_ids[0];
    result.push_back(item);
    return result;
  }
};

TEST_F(ClassificationTestSuite, accuracy) {
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }

  LOGIP("sample_num: %d", results.size());
  int check_num = 0;
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    // printf("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame = loadInputData(image_path);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(cls_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);
    EXPECT_EQ(out_data[0]->getType(), ModelOutputType::CLASSIFICATION);

    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_data[0]);

    if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
      nlohmann::ordered_json result = convertClassificationResult(cls_meta);
      iter.value() = result;
      continue;
    }

    auto expected_cls = iter.value()[0];
    int gt_class_id = expected_cls["class_id"];
    float gt_score = expected_cls["score"];

    int pred_class_id = cls_meta->topk_class_ids[0];
    float pred_score = cls_meta->topk_scores[0];

    EXPECT_EQ(gt_class_id, pred_class_id);
    EXPECT_NEAR(gt_score, pred_score, 0.1);
    check_num++;
  }
  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  } else {
    LOGIP("check_num: %d", check_num);
  }
}

}  // namespace unitest
}  // namespace cvitdl
