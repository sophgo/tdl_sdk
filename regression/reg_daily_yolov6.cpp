#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/cvi_tdl_types_mem.h"
#include "cvi_tdl_log.hpp"
#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "models/tdl_model_factory.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class YoloV6DetectionTestSuite : public CVI_TDLModelTestSuite {
 public:
  YoloV6DetectionTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_yolov6.json", "reg_daily_yolov6") {}

  virtual ~YoloV6DetectionTestSuite() = default;

  std::string m_model_path;
  std::shared_ptr<BaseModel> m_model;
  TDLModelFactory model_factory_;

 protected:
  virtual void SetUp() {
    std::string m_modelname = std::string(m_json_object["model_name"]);
    m_model_path = (m_model_dir / fs::path(m_modelname)).string();
    m_model = model_factory_.getModel(TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV6,
                                      m_model_path);

    ASSERT_NE(m_model, nullptr);
  }

  virtual void TearDown() {}
};

TEST_F(YoloV6DetectionTestSuite, accuracy) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  const float bbox_threshold = 0.5;
  const float score_threshold = 0.4;
  std::cout << "results " << results << std::endl;

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    std::cout << "image_path: " << image_path << std::endl;
    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, true);

    ASSERT_NE(frame, nullptr);
    // break;
    std::vector<void *> out_data;
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);
    EXPECT_EQ(m_model->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1);
    cvtdl_object_t *obj_meta = (cvtdl_object_t *)out_data[0];
    std::cout << "obj_meta->size:" << obj_meta->size << std::endl;
    auto expected_dets = iter.value();
    ASSERT_EQ(obj_meta->size, expected_dets.size());
    std::vector<std::vector<float>> gt_dets;
    for (const auto &det : expected_dets) {
      gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                         det["bbox"][3], det["score"], det["category_id"]});
    }

    std::vector<std::vector<float>> pred_dets;
    for (uint32_t det_index = 0; det_index < obj_meta->size; det_index++) {
      pred_dets.push_back(
          {obj_meta->info[det_index].bbox.x1, obj_meta->info[det_index].bbox.y1,
           obj_meta->info[det_index].bbox.x2, obj_meta->info[det_index].bbox.y2,
           obj_meta->info[det_index].bbox.score,
           float(obj_meta->info[det_index].classes)});
    }

    EXPECT_TRUE(
        matchObjects(gt_dets, pred_dets, bbox_threshold, score_threshold));

    model_factory_.releaseOutput(TDL_MODEL_TYPE_OBJECT_DETECTION_YOLOV6,
                                 out_data);
  }
}

}  // namespace unitest
}  // namespace cvitdl
