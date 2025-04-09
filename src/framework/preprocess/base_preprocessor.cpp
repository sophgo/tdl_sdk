

#include "preprocess/opencv_preprocessor.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
#if not defined(__BM168X__) && not defined(__CMODEL_CVITEK__)
#include "preprocess/vpss_preprocessor.hpp"
#endif

std::vector<float> BasePreprocessor::getRescaleConfig(
    const PreprocessParams& params, const int image_width,
    const int image_height) const {
  std::vector<float> rescale_params;
  float src_w = image_width;
  float src_h = image_height;

  // TODO:check whether need to support crop
  if (params.crop_width != 0 && params.crop_height != 0) {
    src_w = params.crop_width;
    src_h = params.crop_height;
  }
  float scale_x = params.dst_width / src_w;
  float scale_y = params.dst_height / src_h;
  if (params.keep_aspect_ratio) {
    float ratio = std::min(scale_x, scale_y);
    rescale_params.push_back(ratio);
    rescale_params.push_back(ratio);
    int offset_x = (params.dst_width - src_w * ratio) / 2;
    int offset_y = (params.dst_height - src_h * ratio) / 2;
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
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = get_platform();
  }
  LOGI("PreprocessorFactory createPreprocessor,platform:%d\n", platform);
  switch (platform) {
    case InferencePlatform::CVITEK:
    case InferencePlatform::CV186X:
#if not defined(__BM168X__) && not defined(__CMODEL_CVITEK__)
      return std::make_shared<VpssPreprocessor>();
      // return std::make_shared<OpenCVPreprocessor>();
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
    case InferencePlatform::CMODEL_CVITEK:

      return std::make_shared<OpenCVPreprocessor>();

    default:
      return nullptr;
  }
  return nullptr;
}