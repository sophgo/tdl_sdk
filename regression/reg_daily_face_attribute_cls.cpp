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

std::map<TDLObjectAttributeType, std::string> attributes_map = {
    {TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER, "gender"},
    {TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE, "age"},
    {TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK, "mask"},
    {TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_EMOTION, "emotion"},
    {TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES, "glass"}};

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class AttributesTestSuite : public CVI_TDLModelTestSuite {
 public:
  AttributesTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~AttributesTestSuite() = default;

  std::shared_ptr<BaseModel> m_model;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id = std::string(m_json_object["model_id"]);
    m_model = TDLModelFactory::getInstance().getModel(model_id);
    ASSERT_NE(m_model, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(AttributesTestSuite, accuracy) {
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
    EXPECT_EQ(m_model->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::CLS_ATTRIBUTE);

    std::shared_ptr<ModelAttributeInfo> attr_info =
        std::static_pointer_cast<ModelAttributeInfo>(out_data[0]);

    auto expected_info = iter.value();
    std::vector<float> gt_info;
    std::vector<float> pred_info;

    for (auto att_iter = attr_info->attributes.begin();
         att_iter != attr_info->attributes.end(); att_iter++) {
      pred_info.push_back(att_iter->second);
      EXPECT_TRUE(expected_info.count(attributes_map[att_iter->first]));

      gt_info.push_back(expected_info[attributes_map[att_iter->first]]);
    }

    EXPECT_TRUE(matchScore(gt_info, pred_info, score_threshold));
  }
}

}  // namespace unitest
}  // namespace cvitdl