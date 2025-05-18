#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/cvi_tdl_types_mem.h"
#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_factory.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class FaceAttributeClsBmTestSuite : public CVI_TDLModelTestSuite {
 public:
  FaceAttributeClsBmTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_face_attribute_cls.json",
                              "reg_daily_face_attribute_cls") {}

  virtual ~FaceAttributeClsBmTestSuite() = default;

  std::string m_model_path;
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
    det_ = TDLModelFactory::getInstance().getModel(model_id);
    ASSERT_NE(det_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(FaceAttributeClsBmTestSuite, accuracy) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  const float score_threshold = 0.2;

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    std::cout << "image_path: " << image_path << std::endl;
    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, true);

    ASSERT_NE(frame, nullptr);
    // break;
    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(m_model->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    EXPECT_EQ(out_data[0]->getType(), ModelOutputType::ATTRIBUTE);
    std::shared_ptr<ModelAttributeInfo> attr_info =
        std::static_pointer_cast<ModelAttributeInfo>(out_data[0]);
    auto expected_info = iter.value();
    ASSERT_EQ(attr_info->attributes.size(), expected_info.size());
    std::vector<std::vector<float>> gt_info;
    for (const auto &info : expected_info) {
      gt_info.push_back(
          {info["gender_score"], info["age"], info["glass"], info["mask"]});
    }

    std::vector<std::vector<float>> pred_info;
    pred_info.push_back(
        {attr_info->attributes
             [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GENDER],
         attr_info
             ->attributes[TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_AGE],
         attr_info->attributes
             [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_GLASSES],
         attr_info->attributes
             [TDLObjectAttributeType::OBJECT_ATTRIBUTE_HUMAN_MASK]});

    EXPECT_TRUE(matchScore(gt_info, pred_info, score_threshold));
  }
}

}  // namespace unitest
}  // namespace cvitdl