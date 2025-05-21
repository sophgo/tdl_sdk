#include <opencv2/opencv.hpp>

#include "base_image.hpp"

class OpenCVImage : public BaseImage {
 public:
  // 构造与析构
  // OpenCVImage();
  ~OpenCVImage();

  OpenCVImage(uint32_t width, uint32_t height, ImageFormat imageFormat,
              TDLDataType pix_data_type, bool alloc_memory = false,
              std::shared_ptr<BaseMemoryPool> memory_pool = nullptr);

  OpenCVImage(cv::Mat& mat, ImageFormat imageFormat);

  virtual int32_t prepareImageInfo(uint32_t width, uint32_t height,
                                   ImageFormat imageFormat,
                                   TDLDataType pix_data_type,
                                   uint32_t align_size = 0) override;

  virtual int32_t setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                              uint32_t length) override;

  virtual uint32_t getWidth() const override;
  virtual uint32_t getHeight() const override;
  virtual std::vector<uint32_t> getStrides() const override;
  virtual std::vector<uint64_t> getPhysicalAddress() const override;
  virtual std::vector<uint8_t*> getVirtualAddress() const override;
  virtual uint32_t getPlaneNum() const override;

  virtual uint32_t getImageByteSize() const override;

  virtual uint32_t getInternalType() const override;
  virtual void* getInternalData() const override;

 private:
  int convertType(ImageFormat imageFormat, TDLDataType pix_data_type);

 private:
  std::vector<cv::Mat> mats_;  // OpenCV 图像数据
  std::vector<size_t> steps_;
  int32_t mat_type_;     // 图像类型
  uint32_t img_width_;   // 图像宽度
  uint32_t img_height_;  // 图像高度
  char tmp_buffer[1];    // use to construct temp Mat
};
