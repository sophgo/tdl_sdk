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

class SegmentationTestSuite : public CVI_TDLModelTestSuite {
 public:
  SegmentationTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~SegmentationTestSuite() = default;

  std::shared_ptr<BaseModel> det_;

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

TEST_F(SegmentationTestSuite, accuracy) {
  const float mask_threshold = m_json_object["mask_threshold"];

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  auto results = m_json_object[gen_platform()];

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    LOGIP("image_path: %s\n", image_path.c_str());
    printf("image_path: %s\n", image_path.c_str());

    std::shared_ptr<BaseImage> frame =
        ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);

    ASSERT_NE(frame, nullptr);
    std::vector<std::shared_ptr<BaseImage>> input_images;
    input_images.push_back(frame);

    std::vector<std::shared_ptr<ModelOutputInfo>> out_data;
    EXPECT_EQ(det_->inference(input_images, out_data), 0);
    EXPECT_EQ(out_data.size(), 1u);

    ModelOutputType out_type = out_data[0]->getType();
    EXPECT_TRUE(out_type == ModelOutputType::SEGMENTATION ||
                out_type ==
                    ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION);

    auto expected_info = iter.value();
    std::string mask_path =
        (m_image_dir / m_json_object["image_dir"] / expected_info["mask"])
            .string();

    cv::Mat mask_gt = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
    cv::Mat mask_pred;

    if (out_type == ModelOutputType::OBJECT_DETECTION_WITH_SEGMENTATION) {
      std::vector<std::vector<float>> gt_dets;
      std::vector<std::vector<float>> pred_dets;
      std::shared_ptr<ModelBoxSegmentationInfo> obj_meta =
          std::static_pointer_cast<ModelBoxSegmentationInfo>(out_data[0]);
      auto expected_dets = expected_info["detection"];
      for (const auto &det : expected_dets) {
        gt_dets.push_back({det["bbox"][0], det["bbox"][1], det["bbox"][2],
                           det["bbox"][3], det["score"], det["class_id"]});
      }
      for (uint32_t det_index = 0; det_index < obj_meta->box_seg.size();
           det_index++) {
        pred_dets.push_back(
            {obj_meta->box_seg[det_index].x1, obj_meta->box_seg[det_index].y1,
             obj_meta->box_seg[det_index].x2, obj_meta->box_seg[det_index].y2,
             obj_meta->box_seg[det_index].score,
             float(obj_meta->box_seg[det_index].class_id)});
      }

      const float bbox_threshold = m_json_object["bbox_threshold"];
      const float score_threshold = m_json_object["score_threshold"];

      EXPECT_TRUE(
          matchObjects(gt_dets, pred_dets, bbox_threshold, score_threshold));

      int mask_height = obj_meta->mask_height;
      int mask_width = obj_meta->mask_width;

      for (uint32_t i = 0; i < obj_meta->box_seg.size(); i++) {
        cv::Mat src(mask_height, mask_width, CV_8UC1, obj_meta->box_seg[i].mask,
                    mask_width * sizeof(uint8_t));

        if (i == 0) {
          mask_pred = src.clone();
        } else {
          cv::bitwise_or(mask_pred, src, mask_pred);
        }
      }

    } else if (out_type == ModelOutputType::SEGMENTATION) {
      float class_id = 0;
      std::shared_ptr<ModelSegmentationInfo> seg_meta =
          std::static_pointer_cast<ModelSegmentationInfo>(out_data[0]);
      uint32_t output_width = seg_meta->output_width;
      uint32_t output_height = seg_meta->output_height;

      mask_pred = cv::Mat(output_height, output_width, CV_8UC1,
                          seg_meta->class_id, output_width * sizeof(uint8_t));

    } else {
      std::cout << "Unsupported output type: " << static_cast<int>(out_type)
                << std::endl;
      EXPECT_TRUE(0);
      return;
    }

    EXPECT_TRUE(!mask_gt.empty() && !mask_pred.empty());
    EXPECT_TRUE(mask_gt.size() == mask_pred.size());
    EXPECT_TRUE(mask_gt.type() == mask_pred.type());

    EXPECT_TRUE(matchSegmentation(mask_gt, mask_pred, mask_threshold));
  }
}

}  // namespace unitest
}  // namespace cvitdl
