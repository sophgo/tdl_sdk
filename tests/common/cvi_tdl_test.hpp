#pragma once
#include <gtest/gtest.h>
#include <experimental/filesystem>
#include <string>
#include "common/model_output_types.hpp"
#include "image/base_image.hpp"
#include "json.hpp"

namespace fs = std::experimental::filesystem;

namespace cvitdl {
namespace unitest {

enum TestFlag {
  FUNCTION = 0,
  PERFORMANCE,
  GENERATE_FUNCTION_RES,
  GENERATE_PERFORMANCE_RES,
};

class CVI_TDLTestEnvironment : public testing::Environment {
 public:
  explicit CVI_TDLTestEnvironment(const std::string &model_dir,
                                  const std::string &regress_asset_dir,
                                  const std::string &json_file_name);
};

class CVI_TDLTestContext {
 public:
  static CVI_TDLTestContext &getInstance();
  std::experimental::filesystem::path getImageBaseDir();
  std::experimental::filesystem::path getModelBaseDir();
  std::experimental::filesystem::path getJsonFilePath();

  void init(const std::string &model_dir, const std::string &regress_asset_dir,
            const std::string &json_file_name);
  bool setTestFlag(const std::string &test_flag);
  void setTestFlag(TestFlag test_flag) { test_flag_ = test_flag; }
  TestFlag getTestFlag() { return test_flag_; }
  CVI_TDLTestContext(const CVI_TDLTestContext &) = delete;
  CVI_TDLTestContext &operator=(const CVI_TDLTestContext &) = delete;

 private:
  CVI_TDLTestContext();
  ~CVI_TDLTestContext() = default;

  fs::path model_dir_;
  fs::path image_root_;
  fs::path json_root_;
  fs::path json_file_path_;
  TestFlag test_flag_ =
      TestFlag::FUNCTION;  // function,performance,generate_function_res
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
  CVI_TDLModelTestSuite();

  virtual ~CVI_TDLModelTestSuite() = default;

  // object info:[x1,y1,x2,y2,score,class_id]

  std::shared_ptr<BaseImage> loadInputData(std::string &image_path);

 protected:
  /**
   * @brief if test_flag is GENERATE_FUNCTION_RES, check if platform is in
   * json file, if not in, get result from image_dir and if no data inside
   * image_dir return false, if platform is in json file, return false
   * @param test_flag: test flag
   * @param platform: platform
   * @param result: result
   * @return true: success, false: failed
   */
  bool checkToGetProcessResult(TestFlag test_flag, const std::string &platform,
                               nlohmann::ordered_json &result);
  bool generateEmptyJsonFile(const fs::path &json_file_path);
  nlohmann::ordered_json getValidPlatformResult();
  nlohmann::ordered_json getProcessResult();
  void roundFloatPrecision(nlohmann::ordered_json &json_object);
  bool writeJsonFile(const std::string &json_file_path,
                     nlohmann::ordered_json &json_object);

  nlohmann::ordered_json m_json_object;

  // TDLHandle m_tdl_handle;
  fs::path m_image_dir;
  fs::path m_model_dir;
  int float_precesion_num_ = 2;
};

}  // namespace unitest
}  // namespace cvitdl