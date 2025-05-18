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
    cls_ = TDLModelFactory::getInstance().getModel(model_id_);
    ASSERT_NE(cls_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(ClassificationTestSuite, accuracy) {
  const float score_threshold = m_json_object["score_threshold"];

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto results = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    printf("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame = getInputData(image_path, model_id_);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(cls_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    EXPECT_EQ(out_data[0]->getType(), ModelOutputType::CLASSIFICATION);

    std::shared_ptr<ModelClassificationInfo> cls_meta =
        std::static_pointer_cast<ModelClassificationInfo>(out_data[0]);

    auto expected_cls = iter.value()[0];
    int gt_class_id = expected_cls["class_id"];
    float gt_score = expected_cls["score"];

    int pred_class_id = cls_meta->topk_class_ids[0];
    float pred_score = cls_meta->topk_scores[0];

    EXPECT_EQ(gt_class_id, pred_class_id);
    EXPECT_NEAR(gt_score, pred_score, 0.1);
  }
}

}  // namespace unitest
}  // namespace cvitdl
