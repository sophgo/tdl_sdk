#pragma once
#include <experimental/filesystem>
#include <string>
#include "gtest.h"
#include "image/base_image.hpp"
#include "json.hpp"
#include "tdl_model_defs.hpp"
namespace cvitdl {
namespace unitest {

class CVI_TDLTestEnvironment : public testing::Environment {
 public:
  explicit CVI_TDLTestEnvironment(const std::string &model_dir,
                                  const std::string &image_dir,
                                  const std::string &json_dir);
};

class CVI_TDLTestContext {
 public:
  static CVI_TDLTestContext &getInstance();
  std::experimental::filesystem::path getImageBaseDir();
  std::experimental::filesystem::path getModelBaseDir();
  std::experimental::filesystem::path getJsonBaseDir();

  void init(std::string model_dir, std::string image_dir, std::string json_dir);

  CVI_TDLTestContext(const CVI_TDLTestContext &) = delete;
  CVI_TDLTestContext &operator=(const CVI_TDLTestContext &) = delete;

 private:
  CVI_TDLTestContext();
  ~CVI_TDLTestContext() = default;

  std::experimental::filesystem::path m_model_dir;
  std::experimental::filesystem::path m_image_dir;
  std::experimental::filesystem::path m_json_dir;
  bool m_inited;
};

class CVI_TDLTestSuite : public testing::Test {
 public:
  CVI_TDLTestSuite() = default;
  virtual ~CVI_TDLTestSuite() = default;
  static void SetUpTestCase();
  static void TearDownTestCase();
  static int64_t get_ion_memory_size();

 protected:
  static const uint32_t DEFAULT_IMG_WIDTH = 1280;
  static const uint32_t DEFAULT_IMG_HEIGHT = 720;
};

class CVI_TDLModelTestSuite : public CVI_TDLTestSuite {
 public:
  CVI_TDLModelTestSuite(const std::string &json_file_name,
                        const std::string &image_dir_name);
  CVI_TDLModelTestSuite();

  virtual ~CVI_TDLModelTestSuite() = default;

  // object info:[x1,y1,x2,y2,score,class_id]
  bool matchObjects(const std::vector<std::vector<float>> &gt_objects,
                    const std::vector<std::vector<float>> &pred_objects,
                    const float iout_thresh, const float score_thresh);

  bool matchScore(const std::vector<float> &gt_info,
                  const std::vector<float> &pred_info,
                  const float score_thresh);
  ModelType stringToModelType(const std::string &model_type_str);
  std::shared_ptr<BaseImage> getInputData(std::string &image_path,
                                          ModelType model_id);
  bool matchKeypoints(const std::vector<float> &gt_keypoints_x,
                      const std::vector<float> &gt_keypoints_y,
                      const std::vector<float> &gt_keypoints_score,
                      const std::vector<float> &pred_keypoints_x,
                      const std::vector<float> &pred_keypoints_y,
                      const std::vector<float> &pred_keypoints_score,
                      const float position_thresh, const float score_thresh);
  bool matchSegmentation(const cv::Mat &mat1, const cv::Mat &mat2,
                         float mask_thresh);

 protected:
  nlohmann::json m_json_object;
  // TDLHandle m_tdl_handle;
  std::experimental::filesystem::path m_image_dir;
  std::experimental::filesystem::path m_model_dir;
};

}  // namespace unitest
}  // namespace cvitdl