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

class MobileDetectionV2TestSuite : public CVI_TDLModelTestSuite {
 public:
  MobileDetectionV2TestSuite()
      : CVI_TDLModelTestSuite("daily_reg_mobiledet.json",
                              "reg_daily_mobildet") {}

  virtual ~MobileDetectionV2TestSuite() = default;

  std::string m_model_path;

  TDLModelFactory model_factory;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(MobileDetectionV2TestSuite, accuracy) {
  const float bbox_threshold = 0.80;
  const float score_threshold = 0.2;

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    auto results = m_json_object[test_index]["results"];
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    std::string model_path = (m_model_dir / fs::path(model_name)).string();
    std::shared_ptr<BaseModel> m_model =
        model_factory.getModel(ModelType::MBV2_PERSON, model_path);
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
      EXPECT_EQ(out_data[0]->getType(), ModelOutputType::OBJECT_DETECTION);
      std::shared_ptr<ModelBoxInfo> obj_meta =
          std::static_pointer_cast<ModelBoxInfo>(out_data[0]);
      std::cout << "obj_meta->size:" << obj_meta->bboxes.size() << std::endl;
      auto expected_dets = iter.value();
      // ASSERT_EQ(obj_meta->size, expected_dets.size());
      std::vector<std::vector<float>> gt_dets;
      for (const auto &det : expected_dets) {
        gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                           det["bbox"][3], det["score"], det["category_id"]});
      }

      std::vector<std::vector<float>> pred_dets;
      for (uint32_t det_index = 0; det_index < obj_meta->bboxes.size();
           det_index++) {
        pred_dets.push_back(
            {obj_meta->bboxes[det_index].x1, obj_meta->bboxes[det_index].y1,
             obj_meta->bboxes[det_index].x2, obj_meta->bboxes[det_index].y2,
             obj_meta->bboxes[det_index].score,
             float(obj_meta->bboxes[det_index].class_id)});
      }

      EXPECT_TRUE(
          matchObjects(gt_dets, pred_dets, bbox_threshold, score_threshold));
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl
