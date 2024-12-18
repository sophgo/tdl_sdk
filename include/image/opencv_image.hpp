#include <opencv2/opencv.hpp>
#include "base_image.hpp"

class OpenCVImage : public BaseImage {
 public:
  // 构造与析构
  OpenCVImage(const cv::Mat& mat);
  ~OpenCVImage() override = default;

  // 基础属性实现
  int getWidth() const override;
  int getHeight() const override;
  int getChannels() const override;
  uint32_t getFormat() const override;
  void* getData() const override;
  size_t getDataSize() const override;
  std::string getDeviceType() const override;
  void* getPlatformMetadata() const override;

 private:
  cv::Mat mat_;      // OpenCV 图像数据
  uint32_t format_;  // 图像格式
};
