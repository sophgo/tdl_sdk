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
#include "image/vpss_image.hpp"
#include "memory/cvi_memory_pool.hpp"
#include "preprocess/vpss_preprocessor.hpp"
namespace cvitdl {
namespace unitest {

class VpssPreprocessorTestSuite : public CVI_TDLModelTestSuite {
 public:
  VpssPreprocessorTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_vpss_image.json",
                              "reg_daily_preprocess") {}

  virtual ~VpssPreprocessorTestSuite() = default;

 protected:
  void SetUp() override {
    // 测试前的初始化工作
  }

  void TearDown() override {
    // 测试后的清理工作
  }
};

TEST_F(VpssPreprocessorTestSuite, TestPreprocess) {
  // 测试参数
  const uint32_t width = 640;
  const uint32_t height = 480;
  const ImageFormat format = ImageFormat::BGR_PLANAR;
  const ImagePixDataType pixType = ImagePixDataType::UINT8;

  // 创建VPSSImage对象
  std::shared_ptr<VPSSImage> vpssImage =
      std::make_shared<VPSSImage>(width, height, format, pixType);

  EXPECT_EQ(vpssImage->randomFill(), 0);
  std::cout << "vpssImage randfill done" << std::endl;
  PreprocessParams params;
  memset(&params, 0, sizeof(PreprocessParams));

  params.dstWidth = 320;
  params.dstHeight = 240;
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;

  params.dstImageFormat = ImageFormat::BGR_PLANAR;
  params.dstPixDataType = ImagePixDataType::UINT8;

  VpssPreprocessor preprocessor;
  auto dstImage = preprocessor.preprocess(vpssImage, params, nullptr);
  EXPECT_NE(dstImage, nullptr);
}

TEST_F(VpssPreprocessorTestSuite, TestReadPreprocess) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];

  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    std::shared_ptr<VPSSImage> vpssImage = std::make_shared<VPSSImage>();
    int32_t ret = vpssImage->readImage(image_path);
    if (ret != 0) {
      std::cout << "read image failed, ret: " << ret
                << " image_path: " << image_path << std::endl;
      continue;
    }
    std::cout << "read image done, ret: " << ret
              << ",width: " << vpssImage->getWidth()
              << ",height: " << vpssImage->getHeight() << std::endl;
    PreprocessParams params;
    memset(&params, 0, sizeof(PreprocessParams));

    params.dstWidth = 320;
    params.dstHeight = 240;
    params.scale[0] = 1;
    params.scale[1] = 1;
    params.scale[2] = 1;

    ret = vpssImage->writeImage("src.jpg");
    if (ret != 0) {
      std::cout << "write src image failed, ret: " << ret << std::endl;
      continue;
    }
    std::cout << "write src image done, ret: " << ret << std::endl;

    params.dstImageFormat = ImageFormat::BGR_PLANAR;
    params.dstPixDataType = ImagePixDataType::UINT8;

    VpssPreprocessor preprocessor;
    preprocessor.setUseVbPool(true);
    auto dstImage = preprocessor.preprocess(vpssImage, params, nullptr);
    EXPECT_NE(dstImage, nullptr);
    ret = dstImage->writeImage("preprocess.jpg");
    if (ret != 0) {
      std::cout << "write image failed, ret: " << ret << std::endl;
    }
    std::cout << "write image done, ret: " << ret << std::endl;
    break;
  }
}
}  // namespace unitest
}  // namespace cvitdl
