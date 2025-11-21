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
#include "utils/tokenizer_bpe.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class ClipTestSuite : public CVI_TDLModelTestSuite {
 public:
  ClipTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~ClipTestSuite() = default;

  std::shared_ptr<BaseModel> image_model_;
  std::shared_ptr<BaseModel> text_model_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id_image = std::string(m_json_object["model_id_image"]);
    std::string model_id_text = std::string(m_json_object["model_id_text"]);
    std::string model_path_image =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["image_model_name"].get<std::string>() +
        gen_model_suffix();

    std::string model_path_text =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["text_model_name"].get<std::string>() +
        gen_model_suffix();

    image_model_ = TDLModelFactory::getInstance().getModel(
        model_id_image,
        model_path_image);  // One model id may correspond to multiple
                            // models with different sizes
    text_model_ = TDLModelFactory::getInstance().getModel(
        model_id_text,
        model_path_text);  // One model id may correspond to multiple
                           // models with different sizes

    ASSERT_NE(image_model_, nullptr);
    ASSERT_NE(text_model_, nullptr);
  }
  std::string encoder_file = m_image_dir.string() + "/" +
                             std::string(m_json_object["attachment"]) +
                             "/encoder.txt";
  std::string bpe_file = m_image_dir.string() + "/" +
                         std::string(m_json_object["attachment"]) +
                         "/vocab.txt";
  std::string input_file = m_image_dir.string() + "/" +
                           std::string(m_json_object["attachment"]) +
                           "/input.txt";
  virtual void TearDown() {}
};

TEST_F(ClipTestSuite, accuracy) {
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext& context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }
  // 文本特征处理（只做一次）
  std::vector<std::vector<int32_t>> tokens;
  BytePairEncoder bpe(encoder_file, bpe_file);
  int result = bpe.tokenizerBPE(input_file, tokens);

  std::vector<std::shared_ptr<BaseImage>> input_texts;
  for (size_t i = 0; i < tokens.size(); ++i) {
    std::shared_ptr<BaseImage> text = ImageFactory::createImage(
        77, 1, ImageFormat::GRAY, TDLDataType::INT32, true);
    uint8_t* txt_buffer = text->getVirtualAddress()[0];
    memcpy(txt_buffer, tokens[i].data(), 77 * sizeof(int32_t));
    input_texts.push_back(text);
  }

  std::vector<std::shared_ptr<ModelOutputInfo>> out_txt;
  text_model_->inference(input_texts, out_txt);

  std::vector<std::vector<float>> text_features;
  for (size_t i = 0; i < out_txt.size(); i++) {
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_txt[i]);
    std::vector<float> feature_vec(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      feature_vec[j] = feature_ptr[j];
    }
    CommonUtils::normalize(feature_vec);
    text_features.push_back(feature_vec);
  }
  LOGIP("sample_num: %d", results.size());
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();

    std::shared_ptr<BaseImage> frame = loadInputData(image_path);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_image = {frame};
    std::vector<std::shared_ptr<ModelOutputInfo>> out_img;
    image_model_->inference(input_image, out_img);
    std::shared_ptr<ModelFeatureInfo> feature_meta =
        std::static_pointer_cast<ModelFeatureInfo>(out_img[0]);
    std::vector<float> image_feature(feature_meta->embedding_num);
    float* feature_ptr = reinterpret_cast<float*>(feature_meta->embedding);
    for (size_t j = 0; j < feature_meta->embedding_num; j++) {
      image_feature[j] = feature_ptr[j];
    }
    CommonUtils::normalize(image_feature);
    // 计算相似度
    std::vector<float> logits;
    for (size_t j = 0; j < text_features.size(); ++j) {
      float sim = CommonUtils::dot_product(image_feature, text_features[j]);
      logits.push_back(sim * 100.0f);
    }
    std::vector<float> probs = CommonUtils::softmax(logits);

    // 找最大概率对应的文本索引
    size_t max_idx = 0;
    float max_prob = probs[0];
    for (size_t j = 1; j < probs.size(); ++j) {
      if (probs[j] > max_prob) {
        max_prob = probs[j];
        max_idx = j;
      }
    }
    auto expected_cls = iter.value()[0];
    int gt_class_id = expected_cls["class_id"];
    float gt_score = expected_cls["score"];

    int pred_class_id = max_idx;
    float pred_score = max_prob;

    EXPECT_EQ(gt_class_id, pred_class_id);
    EXPECT_NEAR(gt_score, pred_score, 0.1);
  }
}

}  // namespace unitest
}  // namespace cvitdl
