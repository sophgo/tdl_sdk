#include "image/opencv_image.hpp"

#include "memory/cpu_memory_pool.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"

#if defined(__BM168X__) || defined(__CV186X__)
#include "memory/bm_memory_pool.hpp"
#else
#include "memory/cvi_memory_pool.hpp"
#endif

// OpenCVImage::OpenCVImage() {}

OpenCVImage::OpenCVImage(uint32_t width, uint32_t height,
                         ImageFormat imageFormat, TDLDataType pix_data_type,
                         bool alloc_memory,
                         std::shared_ptr<BaseMemoryPool> memory_pool) {
  prepareImageInfo(width, height, imageFormat, pix_data_type);
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  image_type_ = ImageImplType::OPENCV_FRAME;
  memory_pool_ = memory_pool;
  if (memory_pool == nullptr) {
#if defined(__BM168X__) || defined(__CV186X__)
    memory_pool_ = std::make_shared<BmMemoryPool>(nullptr);
#else
    memory_pool_ = std::make_shared<CviMemoryPool>();
#endif
  }
  if (alloc_memory) {
    int32_t ret = allocateMemory();
    if (ret != 0) {
      LOGE("allocateMemory failed");
      throw std::runtime_error("allocateMemory failed");
    }
  }
}

OpenCVImage::OpenCVImage(cv::Mat& mat, ImageFormat imageFormat) {
  if (mat.channels() == 1) {
    assert(imageFormat == ImageFormat::GRAY);
    prepareImageInfo(mat.cols, mat.rows, ImageFormat::GRAY, TDLDataType::UINT8);
  } else if (mat.channels() == 3) {
    assert(imageFormat == ImageFormat::BGR_PACKED ||
           imageFormat == ImageFormat::RGB_PACKED);
    prepareImageInfo(mat.cols, mat.rows, imageFormat, TDLDataType::UINT8);
  } else {
    throw std::runtime_error(
        "OpenCVImage::OpenCVImage(cv::Mat& mat) mat.channels(): " +
        std::to_string(mat.channels()) +
        ", only 1 or 3 channels are supported");
  }
  LOGW(
      "OpenCVImage::OpenCVImage(cv::Mat& mat) mat.channels(): %d, use "
      "BGR_PACKED as default",
      mat.channels());
  image_type_ = ImageImplType::OPENCV_FRAME;
#if defined(__BM168X__) || defined(__CV186X__)
  memory_pool_ = std::make_shared<BmMemoryPool>(nullptr);
#else
  memory_pool_ = std::make_shared<CviMemoryPool>();
#endif
  memory_block_ = std::make_unique<MemoryBlock>();
  memory_block_->physicalAddress = 0;
  memory_block_->virtualAddress = mat.data;
  memory_block_->size = mat.total() * mat.elemSize();
  memory_block_->own_memory = false;

  mats_.clear();
  mats_.push_back(mat);
}

OpenCVImage::~OpenCVImage() {
  if (memory_block_ == nullptr) {
    LOGE("memory_block_ is nullptr");
    return;
  }
  LOGI(
      "to destroy OpenCVImage,width: %d,height: %d,image_format: "
      "%d,pix_data_type: %d,own_memory: "
      "%d",
      img_width_, img_height_, image_format_, pix_data_type_,
      memory_block_->own_memory);
  if (memory_block_ && memory_block_->own_memory) {
    if (memory_pool_ != nullptr) {
      int32_t ret = memory_pool_->release(memory_block_);
      if (ret != 0) {
        LOGE("memory_pool_->release failed");
      }
    } else {
      LOGE("memory_pool_ is nullptr");
      assert(false);
    }
  }
}

int32_t OpenCVImage::convertType(ImageFormat imageFormat,
                                 TDLDataType pix_data_type) {
  if (pix_data_type == TDLDataType::UINT8) {
    if (imageFormat == ImageFormat::RGB_PACKED ||
        imageFormat == ImageFormat::BGR_PACKED) {
      return CV_8UC3;
    } else if (imageFormat == ImageFormat::GRAY ||
               imageFormat == ImageFormat::BGR_PLANAR ||
               imageFormat == ImageFormat::RGB_PLANAR) {
      return CV_8UC1;
    }
  } else if (pix_data_type == TDLDataType::INT8) {
    if (imageFormat == ImageFormat::RGB_PACKED ||
        imageFormat == ImageFormat::BGR_PACKED) {
      return CV_8SC3;
    } else if (imageFormat == ImageFormat::GRAY ||
               imageFormat == ImageFormat::BGR_PLANAR ||
               imageFormat == ImageFormat::RGB_PLANAR) {
      return CV_8SC1;
    }
  } else if (pix_data_type == TDLDataType::FP32) {
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
                                      TDLDataType pix_data_type) {
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
    LOGI("construct planar image");
    for (int i = 0; i < 3; i++) {
      mats_.push_back(cv::Mat(height, width, type, tmp_buffer));
    }
  } else {
    LOGI("Unsupported image format: %d", imageFormat);
    return -1;
  }
  image_format_ = imageFormat;
  pix_data_type_ = pix_data_type;
  img_width_ = width;
  img_height_ = height;
  mat_type_ = type;
  LOGI("CV8SC1: %d,CV8UC1: %d,CV8SC3: %d,CV8UC3: %d", CV_8SC1, CV_8UC1, CV_8SC3,
       CV_8UC3);
  LOGI(
      "image stride: %d,width: %d,height: %d,imgformat: %d,pixformat: "
      "%d,mat_type: %d",
      mats_[0].step[0], width, height, imageFormat, pix_data_type, mat_type_);
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
  uint32_t offset = 0;
  for (const auto& mat : mats_) {
    physical_addresses.push_back(memory_block_->physicalAddress + offset);
    offset += mat.step[0] * mat.rows;
  }
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
  LOGI("setupMemory, phy_addr: %llu, vir_addr: %p, length: %d,num_channels: %d",
       phy_addr, vir_addr, length, mats_.size());

  size_t num_channels = mats_.size();
  steps_.clear();
  mats_.clear();
  uint32_t offset = 0;
  for (size_t i = 0; i < num_channels; i++) {
    int step = img_width_ * CV_ELEM_SIZE(mat_type_);
    steps_.push_back(step);
    int bytes = img_height_ * step;
#ifdef __BM168X__
    mats_.push_back(cv::Mat(img_height_, img_width_, bytes, mat_type_,
                            &steps_[i], vir_addr + offset, phy_addr + offset,
                            -1));
#else
    mats_.push_back(
        cv::Mat(img_height_, img_width_, mat_type_, vir_addr + offset, step));
#endif

    LOGI("update mats_[%d].data: %p,step:%d,matstep:%d", i,
         (void*)mats_[i].data, step, mats_[i].step[0]);
    offset += bytes;
  }

  return 0;
}
