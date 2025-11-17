#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_factory.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class FeatureExtraTestSuite : public CVI_TDLModelTestSuite {
 public:
  FeatureExtraTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~FeatureExtraTestSuite() = default;

  std::shared_ptr<BaseModel> model_;

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
    model_ = TDLModelFactory::getInstance().getModel(
        model_id, model_path);  // One model id may correspond to multiple
                                // models with different sizes

    ASSERT_NE(model_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(FeatureExtraTestSuite, accuracy) {
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }

  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path1 =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    std::shared_ptr<BaseImage> image1 =
        ImageFactory::readImage(image_path1, ImageFormat::RGB_PACKED);
    auto next_iter = std::next(iter);
    nlohmann::ordered_json result;
    for (auto iter2 = next_iter; iter2 != results.end(); iter2++) {
      std::string image_path2 =
          (m_image_dir / m_json_object["image_dir"] / iter2.key()).string();
      LOGIP("compare image: %s vs %s\n",
            fs::path(image_path1).filename().string().c_str(),
            fs::path(image_path2).filename().string().c_str());
      std::shared_ptr<BaseImage> image2 =
          ImageFactory::readImage(image_path2, ImageFormat::RGB_PACKED);
      std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
      std::vector<std::shared_ptr<BaseImage>> input_images = {image1, image2};
      model_->inference(input_images, out_fe);
      std::vector<std::vector<float>> features;
      for (size_t i = 0; i < out_fe.size(); i++) {
        std::shared_ptr<ModelFeatureInfo> feature_meta =
            std::static_pointer_cast<ModelFeatureInfo>(out_fe[i]);

        std::vector<float> feature_vec(feature_meta->embedding_num);
        switch (feature_meta->embedding_type) {
          case TDLDataType::INT8: {
            int8_t *feature_ptr =
                reinterpret_cast<int8_t *>(feature_meta->embedding);
            for (size_t j = 0; j < feature_meta->embedding_num; j++) {
              feature_vec[j] = (float)feature_ptr[j];
            }
            features.push_back(feature_vec);
            break;
          }

          case TDLDataType::UINT8: {
            uint8_t *feature_ptr =
                reinterpret_cast<uint8_t *>(feature_meta->embedding);
            for (size_t j = 0; j < feature_meta->embedding_num; j++) {
              feature_vec[j] = (float)feature_ptr[j];
            }
            features.push_back(feature_vec);
            break;
          }

          case TDLDataType::FP32: {
            float *feature_ptr =
                reinterpret_cast<float *>(feature_meta->embedding);
            for (size_t j = 0; j < feature_meta->embedding_num; j++) {
              feature_vec[j] = (float)feature_ptr[j];
            }
            features.push_back(feature_vec);
            break;
          }
          default:
            assert(false && "Unsupported embedding_type");
        }
      }
      float sim = 0;
      float norm1 = 0;
      float norm2 = 0;
      for (size_t i = 0; i < features[0].size(); i++) {
        sim += features[0][i] * features[1][i];
        norm1 += features[0][i] * features[0][i];
        norm2 += features[1][i] * features[1][i];
      }
      norm1 = sqrt(norm1);
      norm2 = sqrt(norm2);
      float pred_similarity = sim / (norm1 * norm2);
      nlohmann::ordered_json result_item;
      std::string compare_image = fs::path(image_path2).filename().string();
      if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
        result_item["similarity"] = pred_similarity;
        result[compare_image] = result_item;
      } else {
        float gt_similarity = iter.value()[compare_image]["similarity"];
        EXPECT_LT(std::abs(pred_similarity - gt_similarity),
                  reg_score_diff_threshold);
      }
    }
    iter.value() = result;
  }
  if (test_flag == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}  // end of TEST_F

TEST_F(FeatureExtraTestSuite, performance) {
  std::string model_path = m_model_dir.string() + "/" + gen_model_dir() + "/" +
                           m_json_object["model_name"].get<std::string>() +
                           gen_model_suffix();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();
  TestFlag test_flag = context.getTestFlag();
  nlohmann::ordered_json results;
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    return;
  }
  if (results.empty()) {
    LOGIP("performance: no images available, skip");
    return;
  }

  auto iter = results.begin();
  std::string image_path =
      (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
  LOGIP("image_path: %s\n", image_path.c_str());

  std::shared_ptr<BaseImage> frame = loadInputData(image_path);
  ASSERT_NE(frame, nullptr);

  run_performance(model_path, frame, model_);

}  // end of TEST_F
}  // namespace unitest
}  // namespace cvitdl
