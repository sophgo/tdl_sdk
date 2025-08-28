#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_factory.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class FeatureExtraBmTestSuite : public CVI_TDLModelTestSuite {
 public:
  FeatureExtraBmTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_face_feature.json", "reg_daily_fr") {}

  virtual ~FeatureExtraBmTestSuite() = default;

  std::shared_ptr<BaseModel> model_;

 protected:
  virtual void SetUp() {
    TDLModelFactory::getInstance().loadModelConfig();
    TDLModelFactory::getInstance().setModelDir(m_model_dir);
  }

  virtual void TearDown() {}
};

TEST_F(FeatureExtraBmTestSuite, accuracy) {
  const float reg_nms_threshold = 0.85;
  const float reg_score_diff_threshold = 0.2;
  std::cout << "m_json_object: " << m_json_object << std::endl;

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    auto results = m_json_object[test_index]["results"];
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    std::string model_path = (m_model_dir / fs::path(model_name)).string();

    model_ = TDLModelFactory::getInstance().getModel(
        ModelType::RESNET_FEATURE_BMFACE_R34, model_path);
    ASSERT_NE(model_, nullptr);

    std::vector<std::pair<std::string, std::string>> pair_info = {
        {"same_pairs", "same_scores"},
        {"diff_pairs", "diff_scores"},
    };
    auto test_config = m_json_object[test_index];
    for (auto pair_test : pair_info) {
      for (size_t pair_idx = 0; pair_idx < test_config[pair_test.first].size();
           pair_idx++) {
        auto pair = test_config[pair_test.first][pair_idx];
        float expected_score = test_config[pair_test.second][pair_idx];

        std::string image_path1 = (m_image_dir / std::string(pair[0])).string();
        std::string image_path2 = (m_image_dir / std::string(pair[1])).string();
        std::shared_ptr<BaseImage> image1 =
            ImageFactory::readImage(image_path1);
        std::shared_ptr<BaseImage> image2 =
            ImageFactory::readImage(image_path2);
        std::vector<std::shared_ptr<ModelOutputInfo>> out_fe;
        std::vector<std::shared_ptr<BaseImage>> input_images = {image1, image2};
        model_->inference(input_images, out_fe);
        std::vector<std::vector<float>> features;
        for (size_t i = 0; i < out_fe.size(); i++) {
          std::shared_ptr<ModelFeatureInfo> feature =
              std::static_pointer_cast<ModelFeatureInfo>(out_fe[i]);
          std::vector<float> feature_vec(feature->embedding_num);
          float *feature_ptr = (float *)feature->embedding;
          for (size_t j = 0; j < feature->embedding_num; j++) {
            feature_vec[j] = feature_ptr[j];
          }
          features.push_back(feature_vec);
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
        float score = sim / (norm1 * norm2);
        std::cout << "score:" << score << " expected_score:" << expected_score
                  << std::endl;
        EXPECT_LT(std::abs(score - expected_score), 0.9);
      }
    }
  }
}
}  // namespace unitest
}  // namespace cvitdl
