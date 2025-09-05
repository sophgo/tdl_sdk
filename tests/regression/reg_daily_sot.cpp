#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/opencv_image.hpp"
#include "json.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "regression_utils.hpp"
#include "tdl_model_defs.hpp"
#include "tdl_model_factory.hpp"
#include "tracker/tracker_types.hpp"
#include "utils/tdl_log.hpp"

namespace fs = std::experimental::filesystem;
namespace cvitdl {
namespace unitest {

class SotTestSuite : public CVI_TDLModelTestSuite {
 public:
  SotTestSuite() : CVI_TDLModelTestSuite() {}

  virtual ~SotTestSuite() = default;

  std::shared_ptr<BaseModel> sot_;

 protected:
  virtual void SetUp() {
    int32_t ret = TDLModelFactory::getInstance().loadModelConfig();
    if (ret != 0) {
      LOGE("load model config failed");
      return;
    }
    TDLModelFactory::getInstance().setModelDir(m_model_dir);

    std::string model_id = std::string(m_json_object["model_id"]);
    std::string model_path =
        m_model_dir.string() + "/" + gen_model_dir() + "/" +
        m_json_object["model_name"].get<std::string>() + gen_model_suffix();

    sot_ = TDLModelFactory::getInstance().getModel(
        model_id, model_path);  // One model id may correspond to multiple
                                // models with different sizes
    ASSERT_NE(sot_, nullptr);
    sot_->setModelThreshold(m_json_object["model_score_threshold"]);
  }

  nlohmann::ordered_json convertSotResult(const ObjectBoxInfo &box_info) {
    nlohmann::ordered_json result;  // is a list,contains bbox,conf,class_id
    nlohmann::ordered_json item;
    item["bbox"] = {box_info.x1, box_info.y1, box_info.x2, box_info.y2};
    item["score"] = box_info.score;
    item["class_id"] = box_info.class_id;
    result.push_back(item);
    return result;
  }

  virtual void TearDown() {}
};

TEST_F(SotTestSuite, accuracy) {
  const float reg_nms_threshold = m_json_object["reg_nms_threshold"];
  const float reg_score_diff_threshold =
      m_json_object["reg_score_diff_threshold"];

  CVI_TDLTestContext &context = CVI_TDLTestContext::getInstance();

  std::string image_dir = (m_image_dir / m_json_object["image_dir"]).string();
  std::string platform = get_platform_str();
  TestFlag test_flag = CVI_TDLTestContext::getInstance().getTestFlag();
  nlohmann::ordered_json results;
  LOGIP("test_flag: %d", static_cast<int>(test_flag));
  if (!checkToGetProcessResult(test_flag, platform, results)) {
    LOGIP("checkToGetProcessResult failed");
    return;
  }
  size_t sample_num = results.size();
  LOGIP("regression sample num: %d", sample_num);

  std::shared_ptr<Tracker> tracker =
      TrackerFactory::createTracker(TrackerType::TDL_SOT);
  tracker->setModel(sot_);
  std::shared_ptr<BaseImage> image;
  std::vector<std::vector<float>> gt_dets;
  std::vector<std::vector<float>> pred_dets;
  int frame_id = 0;
  for (auto iter = results.begin(); iter != results.end(); iter++) {
    std::string image_path =
        (m_image_dir / m_json_object["image_dir"] / iter.key()).string();
    image = ImageFactory::readImage(image_path, ImageFormat::RGB_PACKED);
    ASSERT_NE(image, nullptr);
    if (frame_id == 0) {
      ObjectBoxInfo init_bbox;
      init_bbox.x1 = 1687.0;
      init_bbox.y1 = 1199.0;
      init_bbox.x2 = 1745.0;
      init_bbox.y2 = 1365.0;
      init_bbox.score = 1.0;
      init_bbox.class_id = 0;
      tracker->initialize(image, {}, init_bbox);
    } else {
      TrackerInfo tracker_info;
      tracker->track(image, frame_id, tracker_info);

      if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
        LOGI("generate function res,image_path: %s\n", image_path.c_str());
        nlohmann::ordered_json result =
            convertSotResult(tracker_info.box_info_);
        iter.value() = result;
        continue;
      }
      nlohmann::ordered_json expected_det = iter.value()[0];
      gt_dets.push_back({expected_det["bbox"][0], expected_det["bbox"][1],
                         expected_det["bbox"][2], expected_det["bbox"][3],
                         expected_det["score"], expected_det["class_id"]});
      pred_dets.push_back({tracker_info.box_info_.x1, tracker_info.box_info_.y1,
                           tracker_info.box_info_.x2, tracker_info.box_info_.y2,
                           tracker_info.box_info_.score,
                           float(tracker_info.box_info_.class_id)});
    }
    frame_id++;
  }
  EXPECT_TRUE(matchObjects(gt_dets, pred_dets, reg_nms_threshold,
                           reg_score_diff_threshold));
  if (context.getTestFlag() == TestFlag::GENERATE_FUNCTION_RES) {
    m_json_object[platform] = results;
    writeJsonFile(context.getJsonFilePath().string(), m_json_object);
  }
}

}  // namespace unitest
}  // namespace cvitdl
