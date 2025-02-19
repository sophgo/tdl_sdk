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

class MobileDetectionV2TestSuite : public CVI_TDLModelTestSuite {
 public:
  MobileDetectionV2TestSuite()
      : CVI_TDLModelTestSuite("daily_reg_mobiledet.json", "reg_daily_mobildet") {}

  virtual ~MobileDetectionV2TestSuite() = default;

  std::string m_model_path;
  std::shared_ptr<BaseModel> m_model;

 protected:
  virtual void SetUp() {
    std::string model_name = std::string(m_json_object["model_name"]);
    m_model_path = (m_model_dir / fs::path(model_name)).string();
    m_model = TDLModelFactory::createModel(
        TDL_MODEL_TYPE_OBJECT_DETECTION_MOBILEDETV2_PEDESTRIAN, m_model_path);

    ASSERT_NE(m_model, nullptr);
  }

  virtual void TearDown() {}
};


TEST_F(MobileDetectionV2TestSuite, accuracy) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  const float bbox_threshold = 0.90;
  const float score_threshold = 0.2;

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

    CVI_TDL_FreeCpp(obj_meta);  // delete expected_res;
    delete obj_meta;
    obj_meta = nullptr;
  }
}





// TEST_F(MobileDetectionV2TestSuite, accuracy) {
//   for (size_t test_index = 0; test_index < m_json_object.size(); test_index++) {
//     std::string model_name = std::string(m_json_object[test_index]["model_name"]);
//     m_model_path = (m_model_dir / fs::path(model_name)).string();

//     ASSERT_EQ(CVI_TDL_OpenModel(m_tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN,
//                                 m_model_path.c_str()),
//               CVI_TDL_SUCCESS);
//     const float bbox_threshold = 0.90;
//     const float score_threshold = 0.1;
//     auto results = m_json_object[test_index]["results"];

//     for (nlohmann::json::iterator iter = results.begin(); iter != results.end(); iter++) {
//       std::string image_path = (m_image_dir / iter.key()).string();
//       Image frame(image_path, PIXEL_FORMAT_RGB_888);
//       ASSERT_TRUE(frame.open());

//       TDLObject<cvtdl_object_t> obj_meta;

//       ASSERT_EQ(CVI_TDL_Detection(m_tdl_handle, frame.getFrame(),
//                                   CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, obj_meta),
//                 CVI_TDL_SUCCESS);

//       auto expected_dets = iter.value();

//       ASSERT_EQ(obj_meta->size, expected_dets.size());

//       for (uint32_t det_index = 0; det_index < expected_dets.size(); det_index++) {
//         auto bbox = expected_dets[det_index]["bbox"];
//         int catId = int(expected_dets[det_index]["category_id"]);

//         cvtdl_bbox_t expected_bbox = {
//             .x1 = float(bbox[0]),
//             .y1 = float(bbox[1]),
//             .x2 = float(bbox[2]),
//             .y2 = float(bbox[3]),
//             .score = float(expected_dets[det_index]["score"]),
//         };

//         auto comp = [=](cvtdl_object_info_t &info, cvtdl_bbox_t &bbox) {
//           if (info.classes == catId && iou(info.bbox, bbox) >= bbox_threshold &&
//               abs(info.bbox.score - bbox.score) <= score_threshold) {
//             return true;
//           }
//           return false;
//         };
//         EXPECT_TRUE(match_dets(*obj_meta, expected_bbox, comp))
//             << "Error!"
//             << "\n"
//             << "expected bbox: (" << expected_bbox.x1 << ", " << expected_bbox.y1 << ", "
//             << expected_bbox.x2 << ", " << expected_bbox.y2 << ")\n"
//             << "score: " << expected_bbox.score << "\n"
//             << "[" << obj_meta->info[det_index].bbox.x1 << "," << obj_meta->info[det_index].bbox.y1
//             << "," << obj_meta->info[det_index].bbox.x2 << "," << obj_meta->info[det_index].bbox.y2
//             << "," << obj_meta->info[det_index].bbox.score << "],\n";
//         // CVI_TDL_FreeCpp(obj_meta);
//       }
//     }
//   }
// }

}  // namespace unitest
}  // namespace cvitdl
