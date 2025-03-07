#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_log.hpp"
#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_factory.hpp"
namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class People_Vehicle_DetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  People_Vehicle_DetectionTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_person_vehicle.json",
                              "reg_daily_person_vehicle") {}

  virtual ~People_Vehicle_DetectionTestSuite() = default;

  std::shared_ptr<BaseModel> model_;
  TDLModelFactory model_factory_;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model_name"]);
    std::string model_path = (m_model_dir / fs::path(model_name)).string();

    model_ = model_factory_.getModel(
        TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE, model_path);

    // std::map<int, int> type_mapping = {{0, 1}, {4, 0}};
    // model_->setTypeMapping(type_mapping);
#ifdef __CV181X__
    model_->setModelThreshold(0.42);
// #elif defined(__BM168X__)
//     model_->setModelThreshold(0.42);
#else
    model_->setModelThreshold(0.4);
#endif
    std::cout << "model threshold:" << model_->getModelThreshold() << std::endl;
    ASSERT_NE(model_, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(People_Vehicle_DetectionTestSuite, accuracy) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  const float bbox_threshold = 0.75;
  const float score_threshold = 0.4;

  int num_processed = 0;
  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    // num_processed += 1;
    // if (num_processed != 6) {
    //   continue;
    // }
    std::string image_path = (m_image_dir / iter.key()).string();
    std::cout << "image_path: " << image_path << std::endl;
    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, true);

    ASSERT_NE(frame, nullptr);
    // break;
    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(model_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    EXPECT_EQ(out_data[0]->getType(), ModelOutputType::OBJECT_DETECTION);
    std::shared_ptr<ModelBoxInfo> obj_meta =
        std::static_pointer_cast<ModelBoxInfo>(out_data[0]);

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

    // break;
  }
}

}  // namespace unitest
}  // namespace cvitdl
