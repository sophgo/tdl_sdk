
#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include "utils/tdl_log.hpp"

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "preprocess/base_preprocessor.hpp"
namespace cvitdl {
namespace unitest {

class PreprocessorTestSuite : public CVI_TDLTestSuite {
 public:
  PreprocessorTestSuite() : CVI_TDLTestSuite() {}

  virtual ~PreprocessorTestSuite() = default;

 protected:
  void SetUp() override {
    // 测试前的初始化工作
    preprocessor_ =
        PreprocessorFactory::createPreprocessor(InferencePlatform::UNKOWN);
    printf("PreprocessorTestSuite SetUp done,address:%p\n",
           preprocessor_.get());
  }

  void TearDown() override {
    // 测试后的清理工作
  }
  std::shared_ptr<BasePreprocessor> preprocessor_;
};

TEST_F(PreprocessorTestSuite, TestPreprocess) {
  // 测试参数
  const uint32_t width = 640;
  const uint32_t height = 480;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const TDLDataType pixType = TDLDataType::UINT8;

  LOGI("TestPreprocess start\n");
  // 创建VPSSImage对象
  std::shared_ptr<BaseImage> image =
      ImageFactory::createImage(width, height, format, pixType, true);
  LOGI("image address:%p\n", image.get());
  EXPECT_NE(image, nullptr);
  LOGI("image type:%d,width:%d,height:%d\n", image->getImageType(),
       image->getWidth(), image->getHeight());
  EXPECT_EQ(image->randomFill(), 0);
  std::cout << "vpssImage randfill done" << std::endl;
  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));

  params.dst_width = 320;
  params.dst_height = 240;
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;

  params.dst_image_format = ImageFormat::BGR_PLANAR;
  params.dst_pixdata_type = TDLDataType::UINT8;

  auto dstImage = preprocessor_->preprocess(image, params, nullptr);
  EXPECT_NE(dstImage, nullptr);

  EXPECT_EQ(dstImage->getWidth(), 320);
  EXPECT_EQ(dstImage->getHeight(), 240);
}

}  // namespace unitest
}  // namespace cvitdl
