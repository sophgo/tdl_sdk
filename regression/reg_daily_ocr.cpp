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

class OcrTestSuite : public CVI_TDLModelTestSuite {
 public:
  OcrTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~OcrTestSuite() = default;

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

TEST_F(OcrTestSuite, accuracy) {
  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto pairs = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = pairs.begin(); iter != pairs.end();
       iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    LOGIP("image_path: %s\n", image_path.c_str());

    auto expected_results = iter.value();
    std::string gt_str = std::string(expected_results["characters"]);

    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;

    model_->inference(input_images, out_data);
    std::vector<std::vector<float>> features;
    for (size_t i = 0; i < out_data.size(); i++) {
      std::shared_ptr<ModelOcrInfo> ocr_meta =
          std::static_pointer_cast<ModelOcrInfo>(out_data[i]);

      std::string pred_str = std::string(ocr_meta->text_info);
      ASSERT_TRUE(gt_str == pred_str);
    }
  }
}
}  // namespace unitest
}  // namespace cvitdl
