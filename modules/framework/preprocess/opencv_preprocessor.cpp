#include "preprocess/opencv_preprocessor.hpp"

#include "cvi_tdl_log.hpp"
#include "image/opencv_image.hpp"

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
  for (int i = 0; i < tmp_bgr.size(); i++) {
    float m = 0, s = 1;
    if (mean.size() > i) m = mean[i];
    if (scale.size() > i) s = scale[i];
    LOGI("mean:%f,scale:%f,srctype:%d,dsttype:%d,addr:%p", m, s,
         tmp_bgr[i].type(), input_channels[i].type(),
         (void*)input_channels[i].data);
    tmp_bgr[i].convertTo(input_channels[i], input_channels[i].type(), s, -m);
  }
}

OpenCVPreprocessor::OpenCVPreprocessor() {}

std::shared_ptr<BaseImage> OpenCVPreprocessor::resize(
    const std::shared_ptr<BaseImage>& image, int newWidth, int newHeight) {
  return nullptr;
}

std::shared_ptr<BaseImage> OpenCVPreprocessor::crop(
    const std::shared_ptr<BaseImage>& image, int x, int y, int width,
    int height) {
  return nullptr;
}

std::shared_ptr<BaseImage> OpenCVPreprocessor::preprocess(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    std::shared_ptr<BaseMemoryPool> memory_pool) {
  return nullptr;
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

  if (src_image->getPlaneNum() == 1 && dst_image->getPlaneNum() == 3) {
    // create temp resized image
    // TODO:use memory pool
    std::vector<float> rescale_params =
        getRescaleConfig(params, src_image->getWidth(), src_image->getHeight());
    int resized_w = src_image->getWidth() * rescale_params[0];
    int resized_h = src_image->getHeight() * rescale_params[1];
    int pad_x = rescale_params[2];
    int pad_y = rescale_params[3];
    cv::Mat tmp_resized = cv::Mat::zeros(params.dstHeight, params.dstWidth,
                                         src_image->getInternalType());
    LOGI("temp_resized type:%d,src_image type:%d", tmp_resized.type(),
         src_image->getInternalType());
    const cv::Mat& src_mat = *(const cv::Mat*)src_image->getInternalData();
    cv::Rect roi(pad_x, pad_y, resized_w, resized_h);

    print_mat(src_mat, "src_mat");
    int flags = cv::INTER_LINEAR;
    if (params.keepAspectRatio) {
      flags = cv::INTER_AREA;  // TODO: when using ROI, only INTER_AREA is
                               // supported,this is a bug of bmopencv
      cv::resize(src_mat, tmp_resized(roi), cv::Size(resized_w, resized_h), 0,
                 0, flags);
    } else {
      cv::resize(src_mat, tmp_resized, cv::Size(resized_w, resized_h), 0, 0,
                 flags);
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
    for (int i = 0; i < dst_image->getPlaneNum(); i++) {
      input_channels.push_back(dsti[i]);
    }
    bgr_split_scale(tmp_resized, tmp_bgr, input_channels, mean, scale, use_rgb);
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
  LOGI("params.dstImageFormat: %d,params.dstPixDataType: %d",
       (int)params.dstImageFormat, (int)params.dstPixDataType);
  std::shared_ptr<OpenCVImage> src_image_ptr = std::make_shared<OpenCVImage>(
      params.dstWidth, params.dstHeight, params.dstImageFormat,
      params.dstPixDataType, false);

  tensor->constructImage(src_image_ptr, batch_idx);
  preprocessToImage(src_image, params, src_image_ptr);
  tensor->flushCache();

  return 0;
}
