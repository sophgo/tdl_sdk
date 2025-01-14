#include "preprocess/opencv_preprocessor.hpp"

#include "cvi_tdl_log.hpp"
#include "image/opencv_image.hpp"
bool compute_pad_resize_param(cv::Size src_size, cv::Size dst_size,
                              std::vector<float>& rescale_params) {
  rescale_params.clear();
  float src_w = src_size.width;
  float src_h = src_size.height;
  float ratio_w = src_w / dst_size.width;
  float ratio_h = src_h / dst_size.height;
  float ratio = std::max(ratio_w, ratio_h);
  // LOG(INFO)<<src_size<<"->"<<dst_size<<",ratio:"<<ratio_w<<","<<ratio_h;
  rescale_params.push_back(ratio);
  rescale_params.push_back(ratio);
  cv::Mat dst_img;
  if (ratio_w != ratio_h) {
    int src_resized_w = lrint(src_w / ratio);
    int src_resized_h = lrint(src_h / ratio);
    int roi_x = (dst_size.width - src_resized_w + 1) / 2;
    int roi_y = (dst_size.height - src_resized_h + 1) / 2;
    // LOG(INFO)<<"scale:"<<ratio<<",src_size:"<<src_resized_w<<","<<src_resized_h<<",roi_xy:"<<roi_x<<","<<roi_y;
    rescale_params.push_back(roi_x);
    rescale_params.push_back(roi_y);
    return true;
  } else {
    rescale_params.push_back(0.0f);
    rescale_params.push_back(0.0f);
    return false;
  }
}

void pad_resize_to_dst(
    const cv::Mat& src_img, cv::Mat& dst_img,
    std::vector<float>& rescale_params, bool keep_aspect_ratio,
    cv::InterpolationFlags inter_flag /*= cv::INTER_NEAREST*/) {
  bool need_pad_resize = false;
  if (keep_aspect_ratio) {
    need_pad_resize = compute_pad_resize_param(src_img.size(), dst_img.size(),
                                               rescale_params);
  }

  if (!need_pad_resize) {
    cv::resize(src_img, dst_img, dst_img.size(), 0, 0, inter_flag);
  } else {
    int src_resized_w = lrint(src_img.cols / rescale_params[0]);
    int src_resized_h = lrint(src_img.rows / rescale_params[1]);
    cv::Rect roi(rescale_params[2], rescale_params[3], src_resized_w,
                 src_resized_h);
    cv::Mat resized_img;
    LOGI("opencv padresize,size:%d,%d,offset:%d,%d", src_resized_w,
         src_resized_h, roi.x, roi.y);
    cv::resize(src_img, dst_img(roi), cv::Size(src_resized_w, src_resized_h), 0,
               0, inter_flag);
  }
}

void bgr_split_scale(const cv::Mat& src_mat, std::vector<cv::Mat>& tmp_bgr,
                     const std::vector<cv::Mat>& input_channels,
                     const std::vector<float>& mean,
                     const std::vector<float>& scale,
                     bool use_rgb /*= false*/) {
  // cv::split is faster than vpp
  cv::split(src_mat, tmp_bgr);
  if (use_rgb) {
    LOGI("use RGB,img size:%d,%d", src_mat.cols, src_mat.rows);
    std::swap(tmp_bgr[0], tmp_bgr[2]);
  }
  for (int i = 0; i < tmp_bgr.size(); i++) {
    float m = 0, s = 1;
    if (mean.size() > i) m = mean[i];
    if (scale.size() > i) s = scale[i];
    LOGI("mean:%f,scale:%f,srctype:%d,dsttype:%d", m, s, tmp_bgr[i].type(),
         input_channels[i].type());
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
    cv::Mat tmp_resized = cv::Mat::zeros(params.dstHeight, params.dstWidth,
                                         src_image->getInternalType());
    std::vector<float> rescale_params;
    const cv::Mat& src_mat = *(const cv::Mat*)src_image->getInternalData();
    print_mat(src_mat, "src_mat");
    compute_pad_resize_param(src_mat.size(), tmp_resized.size(),
                             rescale_params);
    pad_resize_to_dst(src_mat, tmp_resized, rescale_params,
                      params.keepAspectRatio, cv::INTER_LINEAR);
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
    print_mat(dsti[0], "dsti[0]");
    for (int i = 0; i < dst_image->getPlaneNum(); i++) {
      input_channels.push_back(dsti[i]);
    }
    bgr_split_scale(tmp_resized, tmp_bgr, input_channels, mean, scale, use_rgb);
  } else {
    LOGE("dst_image plane num is not 3!\n");
    return -1;
  }

  return 0;
}

int32_t OpenCVPreprocessor::preprocessToTensor(
    const std::shared_ptr<BaseImage>& src_image, const PreprocessParams& params,
    const int batch_idx, std::shared_ptr<BaseTensor> tensor) {
  std::shared_ptr<OpenCVImage> src_image_ptr = std::make_shared<OpenCVImage>();
  int32_t ret = src_image_ptr->prepareImageInfo(
      params.dstWidth, params.dstHeight, params.dstImageFormat,
      params.dstPixDataType);
  if (ret != 0) {
    LOGE("OpenCVImage prepareImageInfo failed!\n");
    return -1;
  }
  tensor->constructImage(src_image_ptr, batch_idx);
  preprocessToImage(src_image, params, src_image_ptr);
  return 0;
}

std::vector<float> OpenCVPreprocessor::getRescaleConfig(
    const PreprocessParams& params, const int image_width,
    const int image_height) {
  std::vector<float> rescale_params;
  if (params.keepAspectRatio) {
    compute_pad_resize_param(cv::Size(image_width, image_height),
                             cv::Size(params.dstWidth, params.dstHeight),
                             rescale_params);
  } else {
    float scale_x = image_width / float(params.dstWidth);
    float scale_y = image_height / float(params.dstHeight);
    rescale_params = {scale_x, scale_y, 0.0f, 0.0f};
  }
  return rescale_params;
}
