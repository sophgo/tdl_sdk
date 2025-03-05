
#include <gtest/gtest.h>

#include <cvi_tdl_log.hpp>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cvi_tdl_test.hpp"
#include "image/base_image.hpp"
#include "preprocess/base_preprocessor.hpp"
namespace cvitdl {
namespace unitest {

class PreprocessorTestSuite : public CVI_TDLModelTestSuite {
 public:
  PreprocessorTestSuite()
      : CVI_TDLModelTestSuite("reg_daily_vpss_image.json",
                              "reg_daily_preprocess") {}

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

  params.dstWidth = 320;
  params.dstHeight = 240;
  params.scale[0] = 1;
  params.scale[1] = 1;
  params.scale[2] = 1;

  params.dstImageFormat = ImageFormat::BGR_PLANAR;
  params.dstPixDataType = TDLDataType::UINT8;

  auto dstImage = preprocessor_->preprocess(image, params, nullptr);
  EXPECT_NE(dstImage, nullptr);
}

TEST_F(PreprocessorTestSuite, TestReadPreprocess) {
  int img_num = int(m_json_object["image_num"]);
  auto results = m_json_object["results"];
  LOGI("TestReadPreprocess start\n");
  for (nlohmann::json::iterator iter = results.begin(); iter != results.end();
       iter++) {
    std::string image_path = (m_image_dir / iter.key()).string();
    std::shared_ptr<BaseImage> image = ImageFactory::readImage(image_path);
    EXPECT_NE(image, nullptr);
    std::cout << "read image done, "
              << ",width: " << image->getWidth()
              << ",height: " << image->getHeight() << std::endl;
    PreprocessParams params;
    memset(&params, 0, sizeof(PreprocessParams));

    params.dstWidth = 320;
    params.dstHeight = 240;
    params.scale[0] = 1;
    params.scale[1] = 1;
    params.scale[2] = 1;

    ImageFormat image_format = ImageFormat::RGB_PLANAR;
    TDLDataType pix_data_type = TDLDataType::UINT8;
    params.dstImageFormat = image_format;
    params.dstPixDataType = pix_data_type;

    auto dstImage = preprocessor_->preprocess(image, params, nullptr);
    EXPECT_NE(dstImage, nullptr);
    EXPECT_EQ(dstImage->getImageType(), ImageImplType::VPSS_FRAME);
    EXPECT_EQ(dstImage->getImageFormat(), image_format);
    EXPECT_EQ(dstImage->getPixDataType(), pix_data_type);
    EXPECT_EQ(dstImage->getWidth(), 320);
    EXPECT_EQ(dstImage->getHeight(), 240);
    int32_t ret = ImageFactory::writeImage("preprocess.jpg", dstImage);

    std::cout << "write image done, ret: " << ret << std::endl;
  }
}
}  // namespace unitest
}  // namespace cvitdl
