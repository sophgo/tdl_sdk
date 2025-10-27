
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#include "cvi_tdl_test.hpp"
#include "tdl_model_factory.hpp"

namespace cvitdl {
namespace unitest {

class NetTestSuite : public CVI_TDLTestSuite {
 public:
  NetTestSuite() : CVI_TDLTestSuite() {}

  virtual ~NetTestSuite() = default;

 protected:
  void SetUp() override {
    // 测试前的初始化工作
    TDLModelFactory::getInstance().loadModelConfig();
    TDLModelFactory::getInstance().setModelDir(
        cvitdl::unitest::CVI_TDLTestContext::getInstance()
            .getModelBaseDir()
            .string());
    model_path_ =
        TDLModelFactory::getInstance().getModelPath(ModelType::SCRFD_DET_FACE);
    EXPECT_NE(model_path_, "");
    std::ifstream model_file(model_path_, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(model_file.is_open());
    std::streampos file_size = model_file.tellg();
    ASSERT_GT(file_size, 0);
    model_buffer_size_ = static_cast<uint32_t>(file_size);
    model_file.seekg(0, std::ios::beg);
    model_buffer_ = new uint8_t[model_buffer_size_];
    model_file.read(reinterpret_cast<char *>(model_buffer_),
                    static_cast<std::streamsize>(model_buffer_size_));
    ASSERT_TRUE(model_file.good());
    model_file.close();
    LOGIP("model_path_: %s", model_path_.c_str());
    LOGIP("model_buffer_size_: %d,address:%p", model_buffer_size_,
          model_buffer_);
  }

  void TearDown() override {
    // 测试后的清理工作
    delete[] model_buffer_;
    model_buffer_ = nullptr;
    model_buffer_size_ = 0;
    model_path_ = "";
  }
  uint8_t *model_buffer_;
  uint32_t model_buffer_size_;
  std::string model_path_;
};

TEST_F(NetTestSuite, TestModelFileLoad) {
  InferencePlatform platform = CommonUtils::getPlatform();
  NetParam net_param_path;
  net_param_path.model_file_path = model_path_;
  std::shared_ptr<BaseNet> net_path =
      NetFactory::createNet(net_param_path, platform);
  EXPECT_NE(net_path, nullptr);
  int32_t ret = net_path->setup();
  EXPECT_EQ(ret, 0);
  NetParam net_param_buffer;
  net_param_buffer.model_buffer = model_buffer_;
  net_param_buffer.model_buffer_size = model_buffer_size_;
  std::shared_ptr<BaseNet> net_buffer =
      NetFactory::createNet(net_param_buffer, platform);
  EXPECT_NE(net_buffer, nullptr);
  ret = net_buffer->setup();
  EXPECT_EQ(ret, 0);

  std::vector<std::string> input_names_path = net_path->getInputNames();
  std::vector<std::string> input_names_buffer = net_buffer->getInputNames();
  EXPECT_EQ(input_names_path.size(), input_names_buffer.size());
  for (int i = 0; i < input_names_path.size(); i++) {
    EXPECT_EQ(input_names_path[i], input_names_buffer[i]);
  }
  std::vector<std::string> output_names_path = net_path->getOutputNames();
  std::vector<std::string> output_names_buffer = net_buffer->getOutputNames();
  EXPECT_EQ(output_names_path.size(), output_names_buffer.size());
  for (int i = 0; i < output_names_path.size(); i++) {
    EXPECT_EQ(output_names_path[i], output_names_buffer[i]);
  }
}

}  // namespace unitest
}  // namespace cvitdl
