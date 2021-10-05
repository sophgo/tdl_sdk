#pragma once
#include <experimental/filesystem>
#include <string>
#include "cviai.h"
#include "gtest.h"
#include "json.hpp"

namespace cviai {
namespace unitest {

class CVIAITestEnvironment : public testing::Environment {
 public:
  explicit CVIAITestEnvironment(const std::string &model_dir, const std::string &image_dir,
                                const std::string &json_dir);
};

class CVIAITestContext {
 public:
  static CVIAITestContext &getInstance();
  std::experimental::filesystem::path getImageBaseDir();
  std::experimental::filesystem::path getModelBaseDir();
  std::experimental::filesystem::path getJsonBaseDir();

  void init(std::string model_dir, std::string image_dir, std::string json_dir);

  CVIAITestContext(const CVIAITestContext &) = delete;
  CVIAITestContext &operator=(const CVIAITestContext &) = delete;

 private:
  CVIAITestContext();
  ~CVIAITestContext() = default;

  std::experimental::filesystem::path m_model_dir;
  std::experimental::filesystem::path m_image_dir;
  std::experimental::filesystem::path m_json_dir;
  bool m_inited;
};

class CVIAITestSuite : public testing::Test {
 public:
  CVIAITestSuite() = default;
  virtual ~CVIAITestSuite() = default;
  static void SetUpTestCase();
  static void TearDownTestCase();
  static int64_t get_ion_memory_size();

 protected:
  static const uint32_t DEFAULT_IMG_WIDTH = 2560;
  static const uint32_t DEFAULT_IMG_HEIGHT = 1440;
};

class CVIAIModelTestSuite : public CVIAITestSuite {
 public:
  CVIAIModelTestSuite(const std::string &json_file_name, const std::string &image_dir_name);

  virtual ~CVIAIModelTestSuite() = default;

 protected:
  nlohmann::json m_json_object;
  cviai_handle_t m_ai_handle;
  std::experimental::filesystem::path m_image_dir;
  std::experimental::filesystem::path m_model_dir;
};

}  // namespace unitest
}  // namespace cviai