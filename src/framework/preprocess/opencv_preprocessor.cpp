#include "preprocess/opencv_preprocessor.hpp"
#include "image/opencv_image.hpp"
#include "utils/tdl_log.hpp"
#if defined(__BM168X__)
#include <opencv2/core/bmcv.hpp>
#endif

void bgr_split_scale(const cv::Mat& src_mat, std::vector<cv::Mat>& tmp_bgr,
                     const std::vector<cv::Mat>& input_channels,
                     const std::vector<float>& mean,
                     const std::vector<float>& scale,
                     bool use_rgb /*= false*/) {
  // cv::split is faster than vpp
  cv::split(src_mat, tmp_bgr);
  if (use_rgb) {
    LOGI("swap RGB,img size:%d,%d", src_mat.cols, src_mat.rows);
    std::swap(tmp_bgr[0], tmp_bgr[2]);
  }
  int src_w = src_mat.cols;
  int src_h = src_mat.rows;
  int dst_w = tmp_bgr[0].cols;
  int dst_h = tmp_bgr[0].rows;
  int pad_x = (dst_w - src_w) / 2;
  int pad_y = (dst_h - src_h) / 2;
  cv::Rect roi(pad_x, pad_y, src_w, src_h);
  for (size_t i = 0; i < tmp_bgr.size(); i++) {
    float m = 0, s = 1;
    if (mean.size() > i) m = mean[i];
    if (scale.size() > i) s = scale[i];
    LOGI("mean:%f,scale:%f,srctype:%d,dsttype:%d,addr:%p,pad_x:%d,pad_y:%d", m,
         s, tmp_bgr[i].type(), input_channels[i].type(),
         (void*)input_channels[i].data, pad_x, pad_y);
    if (pad_x == 0 && pad_y == 0) {
      tmp_bgr[i].convertTo(input_channels[i], input_channels[i].type(), s, -m);
    } else {
      tmp_bgr[i].convertTo(input_channels[i](roi), input_channels[i].type(), s,
                           -m);
    }
  }
}

OpenCVPreprocessor::OpenCVPreprocessor() {}

std::shared_ptr<BaseImage> OpenCVPreprocessor::preprocess(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  if (memory_pool == nullptr) {
    LOGW("input memory_pool is nullptr,use src image memory pool\n");
    memory_pool = src_image->getMemoryPool();
  }
  if (memory_pool == nullptr) {
    LOGE("memory_pool is nullptr!\n");
    return nullptr;
  }

  std::shared_ptr<OpenCVImage> opencv_image = std::make_shared<OpenCVImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false, memory_pool);
  std::unique_ptr<MemoryBlock> memory_block;

  memory_block = memory_pool->allocate(opencv_image->getImageByteSize());
  if (memory_block == nullptr) {
    LOGE("VPSSImage allocate memory failed!\n");
    return nullptr;
  }
  int32_t ret = opencv_image->setupMemoryBlock(memory_block);
  if (ret != 0) {
    LOGE("OpenCVImage setupMemoryBlock failed!\n");
    return nullptr;
  }
  LOGI("setup output image done");

  ret = preprocessToImage(src_image, params, opencv_image);
  if (ret != 0) {
    LOGE("preprocessToImage failed!\n");
    return nullptr;
  }
  LOGI("preprocessToImage done");
  return opencv_image;
}
void print_mat(const cv::Mat& mat, const std::string& name) {
  LOGI("%s: %d,%d,type:%d,addr:%0x", name.c_str(), mat.rows, mat.cols,
       mat.type(), mat.data);
}
int32_t OpenCVPreprocessor::preprocessToImage(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseImage> dst_image) {
  if (!dst_image->isInitialized()) {
    LOGE("dst_image is not initialized!\n");
    return -1;
  }

  std::vector<float> rescale_params =
      getRescaleConfig(params, src_image->getWidth(), src_image->getHeight());

  int pad_x = (params.crop_x - rescale_params[2]) / rescale_params[0];
  int pad_y = (params.crop_y - rescale_params[3]) / rescale_params[1];

  int resized_w = params.dst_width - pad_x * 2;
  int resized_h = params.dst_height - pad_y * 2;
  LOGI("resized_w:%d,resized_h:%d,pad_x:%d,pad_y:%d,src_w:%d,src_h:%d",
       resized_w, resized_h, pad_x, pad_y, src_image->getWidth(),
       src_image->getHeight());
  if (src_image->getPlaneNum() == 1 && dst_image->getPlaneNum() == 3) {
    // create temp resized image
    // TODO:use memory pool

    cv::Mat tmp_resized = cv::Mat::zeros(params.dst_height, params.dst_width,
                                         src_image->getInternalType());
    LOGI("temp_resized type:%d,src_image type:%d", tmp_resized.type(),
         src_image->getInternalType());
    const cv::Mat& src_mat = *(const cv::Mat*)src_image->getInternalData();

    print_mat(src_mat, "src_mat");

    // int flags = cv::INTER_LINEAR;
    // if (params.keep_aspect_ratio) {
    //   flags = cv::INTER_AREA;  // TODO: when using ROI, only INTER_AREA is
    //                            // supported,this is a bug of bmopencv
    //   cv::resize(src_mat, tmp_resized(roi), cv::Size(resized_w, resized_h),
    //   0,
    //              0, flags);
    // } else {
    //   cv::resize(src_mat, tmp_resized, cv::Size(resized_w, resized_h), 0, 0,
    //              flags);
    // }
    if (pad_x != 0 || pad_y != 0) {
      cv::Rect roi(pad_x, pad_y, resized_w, resized_h);

#if defined(__BM168X__)
      if (src_mat.u->addr) {
        std::vector<cv::Mat> src{src_mat};
        std::vector<cv::Rect> srcrect{
            cv::Rect(0, 0, src_mat.cols, src_mat.rows)};
        std::vector<cv::Rect> dstrect{roi};

        cv::bmcv::stitch(src, srcrect, dstrect, tmp_resized);
      } else {
        cv::resize(src_mat, tmp_resized(roi), cv::Size(resized_w, resized_h), 0,
                   0, cv::INTER_LINEAR);
      }

#else

      cv::resize(src_mat, tmp_resized(roi), cv::Size(resized_w, resized_h), 0,
                 0, cv::INTER_LINEAR);
#endif

    } else {
      cv::resize(src_mat, tmp_resized, cv::Size(resized_w, resized_h), 0, 0,
                 cv::INTER_LINEAR);
    }

    bool use_rgb = (src_image->getImageFormat() == ImageFormat::RGB_PACKED &&
                    dst_image->getImageFormat() == ImageFormat::BGR_PLANAR) ||
                   (src_image->getImageFormat() == ImageFormat::BGR_PACKED &&
                    dst_image->getImageFormat() == ImageFormat::RGB_PLANAR);
    std::vector<cv::Mat> tmp_bgr;
    std::vector<cv::Mat> input_channels;
    std::vector<float> mean = {params.mean[0], params.mean[1], params.mean[2]};
    std::vector<float> scale = {params.scale[0], params.scale[1],
                                params.scale[2]};
    cv::Mat* dsti = (cv::Mat*)dst_image->getInternalData();

    LOGI("use_rgb:%d,src_format:%d,dst_format:%d,dst_addr:%p", use_rgb,
         src_image->getImageFormat(), dst_image->getImageFormat(),
         (void*)dsti[0].data);
    print_mat(dsti[0], "dsti[0]");
    for (uint32_t i = 0; i < dst_image->getPlaneNum(); i++) {
      input_channels.push_back(dsti[i]);
    }
    bgr_split_scale(tmp_resized, tmp_bgr, input_channels, mean, scale, use_rgb);
  } else if ((src_image->getPlaneNum() == 1 && dst_image->getPlaneNum() == 1) ||
             (src_image->getPlaneNum() == 3 && dst_image->getPlaneNum() == 3)) {
    if (src_image->getImageFormat() != dst_image->getImageFormat() ||
        src_image->getPixDataType() != dst_image->getPixDataType()) {
      LOGE(
          "src_image and dst_image format:%d,%d and pixdata type:%d,%d are not "
          "same",
          (int)src_image->getImageFormat(), (int)dst_image->getImageFormat(),
          (int)src_image->getPixDataType(), (int)dst_image->getPixDataType());
      return -1;
    }
    cv::Mat* dsti = (cv::Mat*)dst_image->getInternalData();
    cv::Mat* srci = (cv::Mat*)src_image->getInternalData();

    cv::Rect crop_roi(params.crop_x, params.crop_y, params.crop_width,
                      params.crop_height);
    cv::Mat tmp_resized;
    std::vector<float> mean = {params.mean[0], params.mean[1], params.mean[2]};
    std::vector<float> scale = {params.scale[0], params.scale[1],
                                params.scale[2]};
    for (uint32_t i = 0; i < src_image->getPlaneNum(); i++) {
      if (params.crop_width != 0 && params.crop_height != 0) {
        cv::resize(srci[i](crop_roi), tmp_resized,
                   cv::Size(params.dst_width, params.dst_height), 0, 0,
                   cv::INTER_NEAREST);
      } else {
        cv::resize(srci[i], tmp_resized,
                   cv::Size(params.dst_width, params.dst_height), 0, 0,
                   cv::INTER_NEAREST);
      }
      float m = 0, s = 1;
      if (mean.size() > i) m = mean[i];
      if (scale.size() > i) s = scale[i];
      tmp_resized.convertTo(dsti[i], dsti[i].type(), s, -m);
    }
  } else {
    LOGE(
        "image plane num is not supported, src_image plane num: %d, "
        "dst_image plane num: %d",
        src_image->getPlaneNum(), dst_image->getPlaneNum());
    assert(0);
    return -1;
  }

  return 0;
}

int32_t OpenCVPreprocessor::preprocessToTensor(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    const int batch_idx, std::shared_ptr<BaseTensor> tensor) {
  LOGI("params.dst_image_format: %d,params.dst_pixdata_type: %d",
       (int)params.dst_image_format, (int)params.dst_pixdata_type);
  std::shared_ptr<OpenCVImage> src_image_ptr = std::make_shared<OpenCVImage>(
      params.dst_width, params.dst_height, params.dst_image_format,
      params.dst_pixdata_type, false);
  MemoryBlock* M = tensor->getMemoryBlock();
  memset(M->virtualAddress, 0, M->size);
  tensor->constructImage(src_image_ptr, batch_idx);
  preprocessToImage(src_image, params, src_image_ptr);
  tensor->flushCache();

  return 0;
}
