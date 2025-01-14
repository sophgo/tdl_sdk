#include <gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "models/scrfd.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "raii.hpp"
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

  std::string m_model_path;
  SCRFD m_scrfd;

 protected:
  virtual void SetUp() {}

  virtual void TearDown() {}
};

TEST_F(ScrfdDetBmTestSuite, open_close_model) {
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    std::cout << "model_path: " << m_model_path << std::endl;
    ASSERT_EQ(m_scrfd.modelOpen(m_model_path.c_str()), 0);
  }
}

TEST_F(ScrfdDetBmTestSuite, inference) {
  std::shared_ptr<BasePreprocessor> preprocessor =
      std::make_shared<OpenCVPreprocessor>();
  m_scrfd.setPreprocessor(preprocessor);
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    ASSERT_EQ(m_scrfd.modelOpen(m_model_path.c_str()), 0);

    auto results = m_json_object[test_index]["results"];

    std::string image_path = (m_image_dir / results.begin().key()).string();

    {
      std::shared_ptr<OpenCVImage> frame = std::make_shared<OpenCVImage>();
      int32_t ret = frame->readImage(image_path);
      ASSERT_EQ(ret, 0);
      std::vector<void *> out_data;
      std::vector<std::shared_ptr<BaseImage>> input_images;
      input_images.push_back(frame);
      m_scrfd.inference(input_images, out_data);
      //   EXPECT_EQ(m_scrfd.inference(input_images, out_data), 0);
      //   EXPECT_EQ(out_data.size(), 1);
      cvtdl_face_t *face_meta = (cvtdl_face_t *)out_data[0];
      CVI_TDL_FreeCpp(face_meta);
    }
  }
}

TEST_F(ScrfdDetBmTestSuite, accuracy) {
  std::shared_ptr<BasePreprocessor> preprocessor =
      std::make_shared<OpenCVPreprocessor>();
  m_scrfd.setPreprocessor(preprocessor);
  for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
    std::string model_name =
        std::string(m_json_object[test_index]["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();

    ASSERT_EQ(m_scrfd.modelOpen(m_model_path.c_str()), 0);
    const float bbox_threshold = 0.90;
    const float score_threshold = 0.1;
    auto results = m_json_object[test_index]["results"];

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
      EXPECT_EQ(m_scrfd.inference(input_images, out_data), 0);
      EXPECT_EQ(out_data.size(), 1);
      cvtdl_face_t *face_meta = (cvtdl_face_t *)out_data[0];

      auto expected_dets = iter.value();
      ASSERT_EQ(face_meta->size, expected_dets.size());

      for (uint32_t det_index = 0; det_index < expected_dets.size();
           det_index++) {
        auto bbox = expected_dets[det_index]["bbox"];

        cvtdl_bbox_t expected_bbox = {
            .x1 = float(bbox[0]),
            .y1 = float(bbox[1]),
            .x2 = float(bbox[2]),
            .y2 = float(bbox[3]),
            .score = float(expected_dets[det_index]["score"]),
        };

        auto comp = [=](cvtdl_face_info_t &info, cvtdl_bbox_t &bbox) {
          if (iou(info.bbox, bbox) >= bbox_threshold &&
              abs(info.bbox.score - bbox.score) <= score_threshold) {
            return true;
          }
          return false;
        };
        EXPECT_TRUE(match_dets(*face_meta, expected_bbox, comp))
            << "Error!"
            << "\n"
            << "expected bbox: (" << expected_bbox.x1 << ", "
            << expected_bbox.y1 << ", " << expected_bbox.x2 << ", "
            << expected_bbox.y2 << ")\n"
            << "score: " << expected_bbox.score << "\n"
            << "[" << face_meta->info[det_index].bbox.x1 << ","
            << face_meta->info[det_index].bbox.y1 << ","
            << face_meta->info[det_index].bbox.x2 << ","
            << face_meta->info[det_index].bbox.y2 << ","
            << face_meta->info[det_index].bbox.score << "],\n";
      }
      CVI_TDL_FreeCpp(face_meta);
    }
  }
}

}  // namespace unitest
}  // namespace cvitdl
