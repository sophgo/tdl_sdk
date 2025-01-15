#include "image/opencv_image.hpp"

#include "cvi_tdl_log.hpp"
OpenCVImage::OpenCVImage() {}

OpenCVImage::OpenCVImage(uint32_t width, uint32_t height,
                         ImageFormat imageFormat,
                         ImagePixDataType pix_data_type,
                         std::shared_ptr<BaseMemoryPool> memory_pool) {
  prepareImageInfo(width, height, imageFormat, pix_data_type);
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  image_type_ = ImageImplType::OPENCV_FRAME;
  memory_pool_ = memory_pool;
}
int32_t OpenCVImage::convertType(ImageFormat imageFormat,
                                 ImagePixDataType pix_data_type) {
  if (pix_data_type == ImagePixDataType::UINT8) {
    if (imageFormat == ImageFormat::RGB_PACKED ||
        imageFormat == ImageFormat::BGR_PACKED) {
      return CV_8UC3;
    } else if (imageFormat == ImageFormat::GRAY ||
               imageFormat == ImageFormat::BGR_PLANAR ||
               imageFormat == ImageFormat::RGB_PLANAR) {
      return CV_8UC1;
    }
  } else if (pix_data_type == ImagePixDataType::INT8) {
    if (imageFormat == ImageFormat::RGB_PACKED ||
        imageFormat == ImageFormat::BGR_PACKED) {
      return CV_8SC3;
    } else if (imageFormat == ImageFormat::GRAY ||
               imageFormat == ImageFormat::BGR_PLANAR ||
               imageFormat == ImageFormat::RGB_PLANAR) {
      return CV_8SC1;
    }
  } else if (pix_data_type == ImagePixDataType::FP32) {
    if (imageFormat == ImageFormat::GRAY ||
        imageFormat == ImageFormat::BGR_PLANAR ||
        imageFormat == ImageFormat::RGB_PLANAR) {
      return CV_32FC1;
    } else if (imageFormat == ImageFormat::RGB_PACKED ||
               imageFormat == ImageFormat::BGR_PACKED) {
      return CV_32FC3;
    }
  }
  return -1;
}
int32_t OpenCVImage::prepareImageInfo(uint32_t width, uint32_t height,
                                      ImageFormat imageFormat,
                                      ImagePixDataType pix_data_type) {
  int32_t type = convertType(imageFormat, pix_data_type);
  if (type == -1) {
    LOGI("unsupported image format: %d,pix_data_type: %d", imageFormat,
         pix_data_type);
    return -1;
  }
  if (imageFormat == ImageFormat::RGB_PACKED ||
      imageFormat == ImageFormat::BGR_PACKED) {
    mats_.push_back(cv::Mat(height, width, type, tmp_buffer));
  } else if (imageFormat == ImageFormat::GRAY) {
    mats_.push_back(cv::Mat(height, width, type, tmp_buffer));
  } else if (imageFormat == ImageFormat::RGB_PLANAR ||
             imageFormat == ImageFormat::BGR_PLANAR) {
    for (int i = 0; i < 3; i++) {
      mats_.push_back(cv::Mat(height, width, type, tmp_buffer));
    }
  } else {
    LOGI("Unsupported image format: %d", imageFormat);
    return -1;
  }
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  image_type_ = ImageImplType::OPENCV_FRAME;
  img_width_ = width;
  img_height_ = height;
  mat_type_ = type;
  LOGI("CV8SC1: %d,CV8UC1: %d,CV8SC3: %d,CV8UC3: %d", CV_8SC1, CV_8UC1, CV_8SC3,
       CV_8UC3);
  LOGI("image stride: %d,width: %d,height: %d,imgformat: %d,pixformat: %d",
       mats_[0].step[0], width, height, imageFormat, pix_data_type);
  return 0;
}

uint32_t OpenCVImage::getWidth() const { return mats_[0].cols; }
uint32_t OpenCVImage::getHeight() const { return mats_[0].rows; }
std::vector<uint32_t> OpenCVImage::getStrides() const {
  std::vector<uint32_t> strides;
  for (const auto& mat : mats_) {
    strides.push_back(mat.step[0]);
  }
  return strides;
}
std::vector<uint64_t> OpenCVImage::getPhysicalAddress() const {
  std::vector<uint64_t> physical_addresses;

  return physical_addresses;
}
std::vector<uint8_t*> OpenCVImage::getVirtualAddress() const {
  std::vector<uint8_t*> virtual_addresses;
  for (const auto& mat : mats_) {
    virtual_addresses.push_back(mat.data);
  }
  return virtual_addresses;
}
uint32_t OpenCVImage::getPlaneNum() const { return mats_.size(); }

uint32_t OpenCVImage::getInternalType() { return mats_[0].type(); }
void* OpenCVImage::getInternalData() const {
  if (mats_.empty()) {
    return nullptr;
  }
  return (void*)&mats_[0];
}

uint32_t OpenCVImage::getImageByteSize() const {
  if (mats_.empty()) {
    return 0;
  }
  uint32_t size = 0;
  for (const auto& mat : mats_) {
    size += mat.rows * mat.step[0];
  }
  return size;
}

int32_t OpenCVImage::setupMemory(uint64_t phy_addr, uint8_t* vir_addr,
                                 uint32_t length) {
  LOGI("setupMemory, phy_addr: %llu, vir_addr: %p, length: %d", phy_addr,
       vir_addr, length);
  uint8_t* data = (uint8_t*)vir_addr;
  for (size_t i = 0; i < mats_.size(); i++) {
    mats_[i] = cv::Mat(img_height_, img_width_, mat_type_, data);
    data += img_height_ * img_width_ * mat_type_;
  }

  return 0;
}
