#include "image/base_image.hpp"
#if not defined(__BM168X__) && not defined(__CMODEL__)
#include "image/vpss_image.hpp"
#endif

#include <opencv2/opencv.hpp>

#include "image/opencv_image.hpp"
#include "utils/common_utils.hpp"
#include "utils/image_alignment.hpp"
#include "utils/tdl_log.hpp"
std::shared_ptr<BaseImage> ImageFactory::createImage(
    uint32_t width, uint32_t height, ImageFormat imageFormat,
    TDLDataType pixDataType, bool alloc_memory, InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = get_platform();
  }
  LOGI(
      "createImage,width:%d,height:%d,imageFormat:%d,pixDataType:%d,alloc_"
      "memory:%d,platform:%d",
      width, height, imageFormat, pixDataType, alloc_memory, platform);

  if (imageFormat == ImageFormat::GRAY && pixDataType != TDLDataType::UINT8) {
    LOGI("create base image");
    std::shared_ptr<BaseImage> img =
        std::make_shared<BaseImage>(ImageType::RAW_FRAME);
    int32_t ret =
        img->prepareImageInfo(width, height, imageFormat, pixDataType);
    std::shared_ptr<BaseMemoryPool> memory_pool =
        BaseMemoryPoolFactory::createMemoryPool();
    img->setMemoryPool(memory_pool);

    ret = img->allocateMemory();
    if (ret != 0) {
      LOGE("allocateMemory failed");
      return nullptr;
    }
    return img;
  }

  switch (platform) {
    case InferencePlatform::CVITEK:
    case InferencePlatform::CV186X:
    case InferencePlatform::CV184X:
#if not defined(__BM168X__) && not defined(__CMODEL__)
      LOGI("create VPSSImage");
      return std::make_shared<VPSSImage>(width, height, imageFormat,
                                         pixDataType, alloc_memory);
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
    case InferencePlatform::CMODEL:

      LOGI("create OpenCVImage");
      return std::make_shared<OpenCVImage>(width, height, imageFormat,
                                           pixDataType, alloc_memory);
    default:
      return nullptr;
  }
}

std::shared_ptr<BaseImage> ImageFactory::readImage(const std::string& file_path,
                                                   bool use_rgb,
                                                   InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = get_platform();
  }
  cv::Mat img = cv::imread(file_path);
  ImageFormat image_format = ImageFormat::BGR_PACKED;
  if (use_rgb) {
    image_format = ImageFormat::RGB_PACKED;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  }

  if (img.empty()) {
    LOGE("Failed to load image from file: %s", file_path.c_str());
    return nullptr;
  }

  std::shared_ptr<BaseImage> image = ImageFactory::createImage(
      img.cols, img.rows, image_format, TDLDataType::UINT8, false, platform);
  if (image == nullptr) {
    LOGE("Failed to create image");
    return nullptr;
  }
  int32_t ret = image->allocateMemory();
  if (ret != 0) {
    LOGE("Failed to allocate memory");
    return nullptr;
  }
  std::vector<uint8_t*> virtual_addresses = image->getVirtualAddress();
  uint32_t stride = image->getStrides()[0];
  uint8_t* ptr_dst = image->getVirtualAddress()[0];
  uint8_t* ptr_src = img.data;
  if (img.step[0] == stride) {
    memcpy(ptr_dst, ptr_src, img.rows * stride);
  } else {
    for (int r = 0; r < img.rows; r++) {
      uint8_t* dst = ptr_dst + r * stride;
      memcpy(dst, ptr_src + r * img.step[0], img.cols * 3);
    }
  }
  ret = image->flushCache();

  if (ret != 0) {
    LOGE("Failed to flush cache");
    // return nullptr;
  }
  LOGI("read image done,addr:%lx,ptr[100]:%d", image->getVirtualAddress()[0],
       image->getVirtualAddress()[0][100]);
  return image;
}

int32_t ImageFactory::writeImage(const std::string& file_path,
                                 const std::shared_ptr<BaseImage>& image) {
  if (image == nullptr) {
    LOGE("Image is nullptr");
    return -1;
  }
  if (image->getPixDataType() != TDLDataType::UINT8) {
    LOGE("Image pix data type is not UINT8");
    return -1;
  }

  ImageFormat image_format = image->getImageFormat();
  if (image_format != ImageFormat::BGR_PACKED &&
      image_format != ImageFormat::RGB_PACKED &&
      image_format != ImageFormat::GRAY &&
      image_format != ImageFormat::RGB_PLANAR &&
      image_format != ImageFormat::BGR_PLANAR) {
    LOGE("Image format is not supported,format:%d", image_format);
    return -1;
  }
  LOGI("to write image,width:%d,height:%d,format:%d,pix_type:%d,addr:%lx",
       image->getWidth(), image->getHeight(), image_format,
       image->getPixDataType(), image->getVirtualAddress()[0]);
  image->invalidateCache();
  if (image_format == ImageFormat::BGR_PACKED ||
      image_format == ImageFormat::RGB_PACKED) {
    cv::Mat img;
    uint32_t stride = image->getStrides()[0];
    if (stride != image->getWidth() * 3) {
      LOGI(
          "image is not aligned,stride:%d,width:%d,addr:%lx,create a new image",
          stride, image->getWidth(), image->getVirtualAddress()[0]);
      img = cv::Mat::zeros(image->getHeight(), image->getWidth(), CV_8UC3);
      const uint8_t* ptr_src = image->getVirtualAddress()[0];

      uint8_t* ptr_dst = img.data;
      for (int r = 0; r < image->getHeight(); r++) {
        uint8_t* dst = ptr_dst + r * img.step[0];
        memcpy(dst, ptr_src + r * stride, image->getWidth() * 3);
      }

    } else {
      img = cv::Mat(image->getHeight(), image->getWidth(), CV_8UC3,
                    image->getVirtualAddress()[0]);
    }
    if (image_format == ImageFormat::RGB_PACKED) {
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }
    LOGI(
        "write image to "
        "file,file_path:%s,width:%d,height:%d,stride:%d,addr:%lx",
        file_path.c_str(), image->getWidth(), image->getHeight(), stride,
        image->getVirtualAddress()[0]);
    cv::imwrite(file_path, img);
  } else if (image_format == ImageFormat::GRAY) {
    cv::Mat img =
        cv::Mat::zeros(image->getHeight(), image->getWidth(), CV_8UC1);

    std::vector<uint8_t*> virtual_addresses = image->getVirtualAddress();
    uint8_t* ptr_src = virtual_addresses[0];
    std::vector<uint32_t> strides = image->getStrides();
    uint8_t* ptr_dst = img.data;
    for (int r = 0; r < image->getHeight(); r++) {
      uint8_t* dst = ptr_dst + r * img.step[0];
      memcpy(dst, ptr_src + r * strides[0], image->getWidth());
    }
    cv::imwrite(file_path, img);
  } else if (image_format == ImageFormat::RGB_PLANAR ||
             image_format == ImageFormat::BGR_PLANAR) {
    cv::Mat img =
        cv::Mat::zeros(image->getHeight(), image->getWidth(), CV_8UC3);
    std::vector<uint8_t*> virtual_addresses = image->getVirtualAddress();
    std::vector<uint32_t> strides = image->getStrides();
    uint8_t* ptr_dst = img.data;
    for (int r = 0; r < image->getHeight(); r++) {
      uint8_t* dst = ptr_dst + r * img.step[0];
      uint8_t* src_r = virtual_addresses[0] + r * strides[0];
      uint8_t* src_g = virtual_addresses[1] + r * strides[1];
      uint8_t* src_b = virtual_addresses[2] + r * strides[2];
      for (int c = 0; c < image->getWidth(); c++) {
        if (image_format == ImageFormat::RGB_PLANAR) {
          dst[3 * c] = src_b[c];
          dst[3 * c + 1] = src_g[c];
          dst[3 * c + 2] = src_r[c];
        } else {
          dst[3 * c] = src_r[c];
          dst[3 * c + 1] = src_g[c];
          dst[3 * c + 2] = src_b[c];
        }
      }
    }
    if (image_format == ImageFormat::RGB_PLANAR) {
      LOGI("convert rgb  to bgr");
    }
    cv::imwrite(file_path, img);
  }
  return 0;
}

std::shared_ptr<BaseImage> ImageFactory::alignFace(
    const std::shared_ptr<BaseImage>& image, const float* src_landmark_xy,
    const float* dst_landmark_xy, int num_points,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  if (image == nullptr) {
    LOGE("Image is nullptr");
    return nullptr;
  }
  if (image->getImageFormat() != ImageFormat::BGR_PACKED &&
      image->getImageFormat() != ImageFormat::RGB_PACKED) {
    LOGE("only BGR_PACKED or RGB_PACKED format is supported,current format:%d",
         image->getImageFormat());
    return nullptr;
  }
  int dst_img_size = 112;
  std::shared_ptr<BaseImage> aligned_image = ImageFactory::createImage(
      dst_img_size, dst_img_size, image->getImageFormat(), TDLDataType::UINT8,
      false, InferencePlatform::AUTOMATIC);
  if (aligned_image == nullptr) {
    LOGE("Failed to create aligned image");
    return nullptr;
  }

  if (memory_pool != nullptr) {
    aligned_image->setMemoryPool(memory_pool);
  }
  int32_t ret = aligned_image->allocateMemory();
  if (ret != 0) {
    LOGE("Failed to allocate memory");
    return nullptr;
  }

  LOGI("srcimg,width:%d,height:%d,stride:%d,addr:%lx", image->getWidth(),
       image->getHeight(), image->getStrides()[0],
       image->getVirtualAddress()[0]);
  LOGI("dstimg,width:%d,height:%d,stride:%d,addr:%lx", dst_img_size,
       dst_img_size, aligned_image->getStrides()[0],
       aligned_image->getVirtualAddress()[0]);

  tdl_face_warp_affine(image->getVirtualAddress()[0], image->getStrides()[0],
                       image->getWidth(), image->getHeight(),
                       aligned_image->getVirtualAddress()[0],
                       aligned_image->getStrides()[0], dst_img_size,
                       dst_img_size, src_landmark_xy);
  return aligned_image;
}

std::shared_ptr<BaseImage> ImageFactory::alignLicensePlate(
    const std::shared_ptr<BaseImage>& image, const float* src_landmark_xy,
    const float* dst_landmark_xy, int num_points,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  if (image == nullptr) {
    LOGE("Image is nullptr");
    return nullptr;
  }
  if (image->getImageFormat() != ImageFormat::BGR_PACKED &&
      image->getImageFormat() != ImageFormat::RGB_PACKED) {
    LOGE("only BGR_PACKED or RGB_PACKED format is supported,current format:%d",
         image->getImageFormat());
    return nullptr;
  }
  int dst_img_width = 96;
  int dst_img_height = 24;
  std::shared_ptr<BaseImage> aligned_image = ImageFactory::createImage(
      dst_img_width, dst_img_height, image->getImageFormat(), TDLDataType::UINT8,
      false, InferencePlatform::AUTOMATIC);
  if (aligned_image == nullptr) {
    LOGE("Failed to create aligned image");
    return nullptr;
  }

  if (memory_pool != nullptr) {
    aligned_image->setMemoryPool(memory_pool);
  }
  int32_t ret = aligned_image->allocateMemory();
  if (ret != 0) {
    LOGE("Failed to allocate memory");
    return nullptr;
  }

  LOGI("srcimg,width:%d,height:%d,stride:%d,addr:%lx", image->getWidth(),
       image->getHeight(), image->getStrides()[0],
       image->getVirtualAddress()[0]);
  LOGI("dstimg,width:%d,height:%d,stride:%d,addr:%lx", dst_img_width,
       dst_img_height, aligned_image->getStrides()[0],
       aligned_image->getVirtualAddress()[0]);

  tdl_license_plate_warp_affine(image->getVirtualAddress()[0], image->getStrides()[0],
                       image->getWidth(), image->getHeight(),
                       aligned_image->getVirtualAddress()[0],
                       aligned_image->getStrides()[0], dst_img_width,
                       dst_img_height, src_landmark_xy);
  return aligned_image;
}


std::shared_ptr<BaseImage> ImageFactory::wrapVPSSFrame(void* vpss_frame,
                                                       bool own_memory) {
#if not defined(__BM168X__) && not defined(__CMODEL__)
  LOGI("create VPSSImage");
  if (vpss_frame == nullptr) {
    LOGE("vpss_frame is nullptr");
    return nullptr;
  }
  (void)own_memory;
  VIDEO_FRAME_INFO_S* vpss_frame_info = (VIDEO_FRAME_INFO_S*)vpss_frame;
  return std::make_shared<VPSSImage>(*vpss_frame_info);
#else
  LOGI("not support wrapImage on BM168X");
  return nullptr;
#endif
}

std::shared_ptr<BaseImage> ImageFactory::wrapMat(void* mat_frame, bool is_rgb) {
  if (mat_frame == nullptr) {
    LOGE("mat_frame is nullptr");
    return nullptr;
  }
#ifndef NO_OPENCV
  cv::Mat mat = *(cv::Mat*)mat_frame;
  return convertFromMat(mat, is_rgb);
#else
  LOGE("no opencv support");
  return nullptr;
#endif
}
#ifndef NO_OPENCV
std::shared_ptr<BaseImage> ImageFactory::convertFromMat(cv::Mat& mat,
                                                        bool is_rgb) {
  ImageFormat image_format = ImageFormat::BGR_PACKED;
  if (is_rgb) {
    image_format = ImageFormat::RGB_PACKED;
  }
  if (mat.type() != CV_8UC3) {
    LOGE("only RGB_PACK or BGR_PACK image is supported,current type:%d",
         mat.type());
    return nullptr;
  }
  std::shared_ptr<BaseImage> image = ImageFactory::createImage(
      mat.cols, mat.rows, image_format, TDLDataType::UINT8, false);
  if (image == nullptr) {
    LOGE("Failed to create image");
    return nullptr;
  }
  if (image->getImageType() == ImageType::OPENCV_FRAME) {
    image = std::make_shared<OpenCVImage>(mat, image_format);
  } else {
    int32_t ret = image->allocateMemory();
    if (ret != 0) {
      LOGE("Failed to allocate memory");
      return nullptr;
    }
    std::vector<uint8_t*> virtual_addresses = image->getVirtualAddress();
    uint8_t* ptr_dst = virtual_addresses[0];
    uint8_t* ptr_src = mat.data;
    for (int r = 0; r < mat.rows; r++) {
      uint8_t* dst = ptr_dst + r * image->getStrides()[0];
      memcpy(dst, ptr_src + r * mat.step[0], mat.cols * 3);
    }
    ret = image->flushCache();
    if (ret != 0) {
      LOGE("Failed to flush cache");
      return nullptr;
    }
  }
  return image;
}

int32_t ImageFactory::convertToMat(std::shared_ptr<BaseImage>& image,
                                   cv::Mat& mat, bool& is_rgb) {
  if (image == nullptr) {
    LOGE("Image is nullptr");
    return -1;
  }
  if (image->getImageFormat() != ImageFormat::BGR_PACKED &&
      image->getImageFormat() != ImageFormat::RGB_PACKED) {
    LOGE("only BGR_PACKED or RGB_PACKED format is supported,current format:%d",
         image->getImageFormat());
    return -1;
  }

  uint32_t width = image->getWidth();
  uint32_t height = image->getHeight();
  uint32_t stride = image->getStrides()[0];
  uint8_t* ptr_src = image->getVirtualAddress()[0];
  is_rgb = image->getImageFormat() == ImageFormat::RGB_PACKED;

  mat = cv::Mat(height, width, CV_8UC3, ptr_src, stride);
  return 0;
}
#endif