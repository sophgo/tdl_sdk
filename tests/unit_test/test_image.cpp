// #include <gtest/gtest.h>
// #include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <unordered_map>

// #include "core/utils/vpss_helper.h"
// #include "cvi_comm_video.h"
// #include "cvi_sys.h"
// #include "cvi_tdl.h"
// #include "cvi_tdl_evaluation.h"
// #include "cvi_tdl_media.h"
#include <gtest/gtest.h>

#include "cvi_tdl_test.hpp"

#include "image/base_image.hpp"

namespace cvitdl {
namespace unitest {

class ImageTestSuite : public CVI_TDLTestSuite {
 public:
  ImageTestSuite() : CVI_TDLTestSuite() {}

  virtual ~ImageTestSuite() = default;

 protected:
  void SetUp() override {
    // 测试前的初始化工作
  }

  void TearDown() override {
    // 测试后的清理工作
  }
};

TEST_F(ImageTestSuite, TestConstructorAndBasicOperations) {
  // 测试参数
  const uint32_t width = 1920;
  const uint32_t height = 1080;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const TDLDataType pixType = TDLDataType::UINT8;

  std::shared_ptr<BaseImage> image =
      ImageFactory::createImage(width, height, format, pixType, false);

  // 验证基本属性
  EXPECT_EQ(image->getWidth(), width);
  EXPECT_EQ(image->getHeight(), height);
  EXPECT_EQ(image->getImageFormat(), format);
  EXPECT_EQ(image->getPixDataType(), pixType);
}

TEST_F(ImageTestSuite, TestCacheOperations) {
  // 测试参数
  const uint32_t width = 640;
  const uint32_t height = 480;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const TDLDataType pixType = TDLDataType::UINT8;

  // 创建VPSSImage对象
  //   auto memBlock = std::make_unique<MemoryBlock>(width * height * 3);
  std::shared_ptr<BaseImage> image =
      ImageFactory::createImage(width, height, format, pixType, true);

  // 测试缓存操作
  EXPECT_EQ(image->invalidateCache(), 0);
  EXPECT_EQ(image->flushCache(), 0);
}

}  // namespace unitest
}  // namespace cvitdl