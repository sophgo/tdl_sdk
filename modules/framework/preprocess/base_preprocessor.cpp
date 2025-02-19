

#include "cvi_tdl_log.hpp"
#include "preprocess/opencv_preprocessor.hpp"
#include "utils/common_utils.hpp"
#if not defined(__BM168X__)
#include "preprocess/vpss_preprocessor.hpp"
#endif

std::vector<float> BasePreprocessor::getRescaleConfig(
    const PreprocessParams& params, const int image_width,
    const int image_height) const {
  std::vector<float> rescale_params;
  float src_w = image_width;
  float src_h = image_height;

  // TODO:check whether need to support crop
  if (params.cropWidth != 0 && params.cropHeight != 0) {
    src_w = params.cropWidth;
    src_h = params.cropHeight;
  }
  float scale_x = params.dstWidth / src_w;
  float scale_y = params.dstHeight / src_h;
  if (params.keepAspectRatio) {
    float ratio = std::min(scale_x, scale_y);
    rescale_params.push_back(ratio);
    rescale_params.push_back(ratio);
    int offset_x = (params.dstWidth - src_w * ratio) / 2;
    int offset_y = (params.dstHeight - src_h * ratio) / 2;
    rescale_params.push_back(offset_x);
    rescale_params.push_back(offset_y);
  } else {
    rescale_params.push_back(scale_x);
    rescale_params.push_back(scale_y);
    rescale_params.push_back(0.0f);
    rescale_params.push_back(0.0f);
  }

  return rescale_params;
}
std::shared_ptr<BasePreprocessor> PreprocessorFactory::createPreprocessor(
    InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN) {
    platform = get_platform();
  }
  LOGI("PreprocessorFactory createPreprocessor,platform:%d\n", platform);
  switch (platform) {
    case InferencePlatform::CVITEK:
    case InferencePlatform::CV186X:
#if not defined(__BM168X__)
      return std::make_shared<VpssPreprocessor>();
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
#if defined(__BM168X__)
      return std::make_shared<OpenCVPreprocessor>();
#else
      return nullptr;
#endif
  }
  return nullptr;
}