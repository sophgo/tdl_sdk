#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/cvi_tdl_types_mem.h"
#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "models/tdl_model_factory.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class People_Vehicle_DetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  People_Vehicle_DetectionTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_person_vehicle.json",
                              "reg_daily_person_vehicle") {}

  virtual ~People_Vehicle_DetectionTestSuite() = default;

  std::string m_model_path;
  std::shared_ptr<BaseModel> m_model;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    m_model = TDLModelFactory::createModel(
        TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV8_PERSON_VEHICLE, m_model_path);
    // std::shared_ptr<BasePreprocessor> preprocessor =
    //     std::make_shared<OpenCVPreprocessor>();
    // m_model->setPreprocessor(preprocessor);
    std::map<int, int> type_mapping = {{0, 1}, {4, 0}};
    m_model->setTypeMapping(type_mapping);
    ASSERT_NE(m_model, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(People_Vehicle_DetectionTestSuite, inference) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];

  std::string image_path = (m_image_dir / results.begin().key()).string();
  {
    std::shared_ptr<OpenCVImage> frame = std::make_shared<OpenCVImage>();
    int32_t ret = frame->readImage(image_path);
    ASSERT_EQ(ret, 0);
    std::vector<void *> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(m_model->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_data[0];
    CVI_TDL_FreeCpp(obj_meta);
  }
}

TEST_F(People_Vehicle_DetectionTestSuite, accuracy) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  const float bbox_threshold = 0.85;
  const float score_threshold = 0.2;

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    std::cout << "image_path: " << image_path << std::endl;
    std::shared_ptr<OpenCVImage> frame = std::make_shared<OpenCVImage>();
    int32_t ret = frame->readImage(image_path);
    ASSERT_EQ(ret, 0);
    std::vector<void *> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(m_model->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_data[0];

    auto expected_dets = iter.value();
    ASSERT_EQ(obj_meta->size, expected_dets.size());

    for (uint32_t det_index = 0; det_index < expected_dets.size();
         det_index++) {
      auto bbox = expected_dets[det_index]["bbox"];
      int catId = int(expected_dets[det_index]["category_id"]);

      cvtdl_bbox_t expected_bbox = {
          .x1 = float(bbox[0]),
          .y1 = float(bbox[1]),
          .x2 = float(bbox[2]),
          .y2 = float(bbox[3]),
          .score = float(expected_dets[det_index]["score"]),
      };

      auto comp = [=](cvtdl_object_info_t &info, cvtdl_bbox_t &bbox) {
        if (info.classes == catId && iou(info.bbox, bbox) >= bbox_threshold &&
            abs(info.bbox.score - bbox.score) <= score_threshold) {
          return true;
        }
        return false;
      };
      EXPECT_TRUE(match_dets(*obj_meta, expected_bbox, comp))
          << "Error!"
          << "\n"
          << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1
          << ", " << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n"
          << "score: " << expected_bbox.score << ",label:" << catId << "\n"
          << "[" << obj_meta->info[det_index].bbox.x1 << ","
          << obj_meta->info[det_index].bbox.y1 << ","
          << obj_meta->info[det_index].bbox.x2 << ","
          << obj_meta->info[det_index].bbox.y2 << ","
          << obj_meta->info[det_index].classes << ","
          << obj_meta->info[det_index].bbox.score << "],\n";
    }
    CVI_TDL_FreeCpp(obj_meta);  // delete expected_res;
  }
}

}  // namespace unitest
}  // namespace cvitdl
