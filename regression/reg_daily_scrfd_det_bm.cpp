#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/cvi_tdl_types_mem.h"
#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "json.hpp"
#include "models/tdl_model_factory.hpp"
#include "preprocess/base_preprocessor.hpp"
#include "regression_utils.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class ScrfdDetBmTestSuite : public CVI_TDLModelTestSuite {
 public:
  ScrfdDetBmTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_scrfdface.json",
                              "reg_daily_scrfdface") {}

  virtual ~ScrfdDetBmTestSuite() = default;

  std::shared_ptr<BaseModel> scrfd_;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(ScrfdDetBmTestSuite, accuracy) {
  const float bbox_threshold = 0.85;
  const float score_threshold = 0.2;
  std::cout << "m_json_object: " << m_json_object << std::endl;

  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    auto results = m_json_object[test_index]["results"];
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    std::string model_path = (m_model_dir / fs::path(model_name)).string();
    scrfd_ = TDLModelFactory::createModel(TDL_MODEL_TYPE_FACE_DETECTION_SCRFD,
                                          model_path);
    ASSERT_NE(scrfd_, nullptr);
    for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
         iter++) {
      std::string image_path = (m_image_dir / iter.key()).string();
      std::cout << "image_path: " << image_path << std::endl;
      std::shared_ptr<BaseImage> frame =
          ImageFactory::readImage(image_path, true);
      ASSERT_NE(frame, nullptr);
      std::vector<void *> out_data;
      std::vector<std::shared_ptr<BaseImage>> input_images;
      input_images.push_back(frame);
      EXPECT_EQ(scrfd_->inference(input_images, out_data), 0);
      EXPECT_EQ(out_data.size(), 1);
      cvtdl_face_t *face_meta = (cvtdl_face_t *)out_data[0];

      auto expected_dets = iter.value();
      std::vector<std::vector<float>> gt_dets;
      for (const auto &det : expected_dets) {
        gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                           det["bbox"][3], det["score"], 1});
      }

      std::vector<std::vector<float>> pred_dets;
      for (uint32_t det_index = 0; det_index < face_meta->size; det_index++) {
        pred_dets.push_back({face_meta->info[det_index].bbox.x1,
                             face_meta->info[det_index].bbox.y1,
                             face_meta->info[det_index].bbox.x2,
                             face_meta->info[det_index].bbox.y2,
                             face_meta->info[det_index].bbox.score, 1});
      }

      EXPECT_TRUE(
          matchObjects(gt_dets, pred_dets, bbox_threshold, score_threshold));

      CVI_TDL_FreeCpp(face_meta);
    }
  }
}
}  // namespace unitest
}  // namespace cvitdl
