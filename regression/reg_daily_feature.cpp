#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_log.hpp"
#include "tdl_model_factory.hpp"

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
  const float score_threshold = m_json_object["score_threshold"];
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto pairs = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = pairs.begin(); iter != pairs.end();
       iter++) {
    auto pair_info = iter.value();
    auto image_pair = pair_info["images"];
    float gt_similarity = pair_info["similarity"];

    std::string image_path1 =
        (m_image_dir / m_json_object["image_dir"] / image_pair[0]).string();
    std::string image_path2 =
        (m_image_dir / m_json_object["image_dir"] / image_pair[1]).string();
    std::shared_ptr<BaseImage> image1 = ImageFactory::readImage(image_path1);
    std::shared_ptr<BaseImage> image2 = ImageFactory::readImage(image_path2);
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
    std::cout << "pred_similarity:" << pred_similarity
              << " gt_similarity:" << gt_similarity << std::endl;
    EXPECT_LT(std::abs(pred_similarity - gt_similarity), score_threshold);
  }
}
}  // namespace unitest
}  // namespace cvitdl
