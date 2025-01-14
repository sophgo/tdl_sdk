// #include <gtest.h>
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
#include "image/vpss_image.hpp"
// #include "json.hpp"
#include "memory/cvi_memory_pool.hpp"

namespace cvitdl {
namespace unitest {

class VPSSImageTestSuite : public CVI_TDLModelTestSuite {
 public:
  VPSSImageTestSuite() : CVI_TDLModelTestSuite("", "reg_daily_vpss") {}

  virtual ~VPSSImageTestSuite() = default;

 protected:
  void SetUp() override {
    // 测试前的初始化工作
  }

  void TearDown() override {
    // 测试后的清理工作
  }
};

TEST_F(VPSSImageTestSuite, TestConstructorAndBasicOperations) {
  // 测试参数
  const uint32_t width = 1920;
  const uint32_t height = 1080;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const ImagePixDataType pixType = ImagePixDataType::UINT8;

  // 创建VPSSImage对象
  VPSSImage vpssImage(width, height, format, pixType);

  // 验证基本属性
  EXPECT_EQ(vpssImage.getWidth(), width);
  EXPECT_EQ(vpssImage.getHeight(), height);
  EXPECT_EQ(vpssImage.getImageFormat(), format);
  EXPECT_EQ(vpssImage.getPixDataType(), pixType);
}

TEST_F(VPSSImageTestSuite, TestCacheOperations) {
  // 测试参数
  const uint32_t width = 640;
  const uint32_t height = 480;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const ImagePixDataType pixType = ImagePixDataType::UINT8;

  // 创建VPSSImage对象
  //   auto memBlock = std::make_unique<MemoryBlock>(width * height * 3);
  VPSSImage vpssImage(width, height, format, pixType);

  // 测试缓存操作
  EXPECT_EQ(vpssImage.invalidateCache(), 0);
  EXPECT_EQ(vpssImage.flushCache(), 0);
}

TEST_F(VPSSImageTestSuite, TestPixelFormatConversion) {
  const uint32_t width = 640;
  const uint32_t height = 480;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const ImagePixDataType pixType = ImagePixDataType::UINT8;

  VPSSImage vpssImage(width, height, format, pixType);

  // 测试像素格式转换
  PIXEL_FORMAT_E pixelFormat = vpssImage.convertPixelFormat(format, pixType);
  EXPECT_EQ(pixelFormat, PIXEL_FORMAT_BGR_888_PLANAR);
}

// TEST_F(VPSSImageTestSuite, TestMemoryManagement) {
//   const uint32_t width = 640;
//   const uint32_t height = 480;
//   const ImageFormat format = ImageFormat::BGR_PLANAR;
//   const ImagePixDataType pixType = ImagePixDataType::UINT8;

//   // 使用Mock内存块
//   // auto memPool = std::make_unique<CviMemoryPool>();
//   // auto mockMemBlock = memPool->allocate(width * height * 3);
//   // VPSSImage vpssImage(width, height, format, pixType,
//   // std::move(mockMemBlock)); EXPECT_EQ(vpssImage.getImageByteSize(), width
//   *
//   // height * 3);
// }

}  // namespace unitest
}  // namespace cvitdl