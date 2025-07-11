#include <opencv2/opencv.hpp>

#include "image/base_image.hpp"
#include "image/opencv_image.hpp"
#include "utils/common_utils.hpp"
#include "utils/image_alignment.hpp"
#include "utils/tdl_log.hpp"
#if not defined(__BM168X__) && not defined(__CMODEL_CV181X__) && \
    not defined(__CMODEL_CV184X__)
#include "image/vpss_image.hpp"
#endif
#if defined(__BM168X__)
#include "image/bmcv_image.hpp"
#endif
std::shared_ptr<BaseImage> ImageFactory::createImage(
    uint32_t width, uint32_t height, ImageFormat imageFormat,
    TDLDataType pixDataType, bool alloc_memory, InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = CommonUtils::getPlatform();
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
#if not defined(__BM168X__) && not defined(__CMODEL_CV181X__) && \
    not defined(__CMODEL_CV184X__)
      LOGI("create VPSSImage");
      return std::make_shared<VPSSImage>(width, height, imageFormat,
                                         pixDataType, alloc_memory);
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
#if defined(USE_BMCV)
      LOGI("create BmCVImage");
      return std::make_shared<BmCVImage>(width, height, imageFormat,
                                         pixDataType, alloc_memory);
#else
      LOGI("create OpenCVImage");
      return std::make_shared<OpenCVImage>(width, height, imageFormat,
                                           pixDataType, alloc_memory);
#endif
    case InferencePlatform::CMODEL_CV181X:
    case InferencePlatform::CMODEL_CV184X:
      LOGI("create OpenCVImage");
      return std::make_shared<OpenCVImage>(width, height, imageFormat,
                                           pixDataType, alloc_memory);
    default:
      return nullptr;
  }
}

std::shared_ptr<BaseImage> ImageFactory::readImage(const std::string& file_path,
                                                   ImageFormat image_format,
                                                   InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = CommonUtils::getPlatform();
  }
  cv::Mat img = cv::imread(file_path);
  if (img.empty()) {
    LOGE("Failed to load image from file: %s", file_path.c_str());
    return nullptr;
  }

  std::shared_ptr<BaseImage> image;

  switch (image_format) {
    case ImageFormat::BGR_PACKED:
      image = ImageFactory::createImage(img.cols, img.rows, image_format,
                                        TDLDataType::UINT8, false, platform);
      break;

    case ImageFormat::RGB_PACKED:
      // 从BGR转换到RGB
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      image = ImageFactory::createImage(img.cols, img.rows, image_format,
                                        TDLDataType::UINT8, false, platform);
      break;

    case ImageFormat::GRAY:
      // 转换为灰度图
      cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
      image = ImageFactory::createImage(img.cols, img.rows, image_format,
                                        TDLDataType::UINT8, false, platform);
      break;

    default:
      LOGE("Unsupported image format: %d", static_cast<int>(image_format));
      return nullptr;
  }

  if (image == nullptr) {
    LOGE("Failed to create image");
    return nullptr;
  }

  int32_t ret = image->allocateMemory();
  if (ret != 0) {
    LOGE("Failed to allocate memory");
    return nullptr;
  }

  // 根据不同格式复制数据
  if (image_format == ImageFormat::BGR_PACKED ||
      image_format == ImageFormat::RGB_PACKED) {
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
  } else if (image_format == ImageFormat::GRAY) {
    uint32_t stride = image->getStrides()[0];
    uint8_t* ptr_dst = image->getVirtualAddress()[0];
    uint8_t* ptr_src = img.data;

    for (int r = 0; r < img.rows; r++) {
      uint8_t* dst = ptr_dst + r * stride;
      memcpy(dst, ptr_src + r * img.step[0], img.cols);
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
    LOGE("Image format is not supported,format:%d",
         static_cast<int>(image_format));
    return -1;
  }
  LOGI("to write image,width:%d,height:%d,format:%d,pix_type:%d,addr:%lx",
       image->getWidth(), image->getHeight(), image_format,
       image->getPixDataType(), image->getVirtualAddress()[0]);
  image->invalidateCache();
  if (image_format == ImageFormat::BGR_PACKED ||
      image_format == ImageFormat::RGB_PACKED) {
    cv::Mat img;
    img = cv::Mat(image->getHeight(), image->getWidth(), CV_8UC3,
                  image->getVirtualAddress()[0], image->getStrides()[0]);
    if (image_format == ImageFormat::RGB_PACKED) {
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }
    cv::imwrite(file_path, img);
  } else if (image_format == ImageFormat::GRAY) {
    cv::Mat img(image->getHeight(), image->getWidth(), CV_8UC1,
                image->getVirtualAddress()[0], image->getStrides()[0]);
    cv::imwrite(file_path, img);
  } else if (image_format == ImageFormat::RGB_PLANAR ||
             image_format == ImageFormat::BGR_PLANAR) {
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; i++) {
      channels.push_back(cv::Mat(image->getHeight(), image->getWidth(), CV_8UC1,
                                 image->getVirtualAddress()[i],
                                 image->getStrides()[i]));
    }
    // 合并通道
    cv::Mat img;
    if (image_format == ImageFormat::RGB_PLANAR) {
      // RGB -> BGR
      std::vector<cv::Mat> bgr_channels = {channels[2], channels[1],
                                           channels[0]};
      cv::merge(bgr_channels, img);
      LOGI("convert rgb to bgr");
    } else {
      cv::merge(channels, img);
    }
    cv::imwrite(file_path, img);
  }
  LOGI(
      "write image to "
      "file,file_path:%s,width:%d,height:%d,stride:%d,addr:%lx",
      file_path.c_str(), image->getWidth(), image->getHeight(),
      image->getStrides()[0], image->getVirtualAddress()[0]);
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
      image->getImageFormat() != ImageFormat::RGB_PACKED &&
      image->getImageFormat() != ImageFormat::YUV420SP_VU) {
    LOGE(
        "only BGR_PACKED or RGB_PACKED or YUV420SP_VU format is "
        "supported,current format:%d",
        static_cast<int>(image->getImageFormat()));
    return nullptr;
  }
  int dst_img_size = 112;
  ImageFormat dst_format = image->getImageFormat() == ImageFormat::YUV420SP_VU
                               ? ImageFormat::RGB_PACKED
                               : image->getImageFormat();
  std::shared_ptr<BaseImage> aligned_image = ImageFactory::createImage(
      dst_img_size, dst_img_size, dst_format, TDLDataType::UINT8, false,
      InferencePlatform::AUTOMATIC);
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

  LOGI("dstimg,width:%d,height:%d,stride:%d,addr:%lx", dst_img_size,
       dst_img_size, aligned_image->getStrides()[0],
       aligned_image->getVirtualAddress()[0]);

  if (image->getImageFormat() == ImageFormat::YUV420SP_VU) {
    uint32_t height = image->getHeight();
    uint32_t width = image->getWidth();

    size_t nv21_size = image->getHeight() * image->getWidth() * 3 / 2;
    unsigned char* nv21_data = (unsigned char*)malloc(nv21_size);

    memcpy(nv21_data, image->getVirtualAddress()[0], width * height);
    memcpy(nv21_data + width * height, image->getVirtualAddress()[1],
           height * width / 2);

    cv::Mat nv21_img(height + height / 2, width, CV_8UC1, nv21_data);

    cv::Mat rgb_img;
    cv::cvtColor(nv21_img, rgb_img, cv::COLOR_YUV2RGB_NV21);

    LOGI("srcimg,width:%d,height:%d,stride:%d,addr:%lx", height, width,
         rgb_img.step, rgb_img.data);

    tdl_face_warp_affine(rgb_img.data, rgb_img.step, width, height,
                         aligned_image->getVirtualAddress()[0],
                         aligned_image->getStrides()[0], dst_img_size,
                         dst_img_size, src_landmark_xy);
    return aligned_image;
  }

  LOGI("srcimg,width:%d,height:%d,stride:%d,addr:%lx", image->getWidth(),
       image->getHeight(), image->getStrides()[0],
       image->getVirtualAddress()[0]);

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
         static_cast<int>(image->getImageFormat()));
    return nullptr;
  }
  int dst_img_width = 96;
  int dst_img_height = 24;
  std::shared_ptr<BaseImage> aligned_image = ImageFactory::createImage(
      dst_img_width, dst_img_height, image->getImageFormat(),
      TDLDataType::UINT8, false, InferencePlatform::AUTOMATIC);
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

  tdl_license_plate_warp_affine(
      image->getVirtualAddress()[0], image->getStrides()[0], image->getWidth(),
      image->getHeight(), aligned_image->getVirtualAddress()[0],
      aligned_image->getStrides()[0], dst_img_width, dst_img_height,
      src_landmark_xy);
  return aligned_image;
}

std::shared_ptr<BaseImage> ImageFactory::wrapVPSSFrame(void* vpss_frame,
                                                       bool own_memory) {
#if not defined(__BM168X__) && not defined(__CMODEL_CV181X__) && \
    not defined(__CMODEL_CV184X__)
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
         static_cast<int>(image->getImageFormat()));
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