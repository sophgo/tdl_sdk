

#include "preprocess/opencv_preprocessor.hpp"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
#if not defined(__BM168X__) && not defined(__CMODEL_CV181X__) && \
    not defined(__CMODEL_CV184X__)
#include "preprocess/vpss_preprocessor.hpp"
#endif

std::vector<float> BasePreprocessor::getRescaleConfig(
    const PreprocessParams& params, const int image_width,
    const int image_height) const {
  // 1) determine the crop region (or full image if no crop)
  int cx = params.crop_x;
  int cy = params.crop_y;
  int cw = (params.crop_width > 0 ? params.crop_width : image_width);
  int ch = (params.crop_height > 0 ? params.crop_height : image_height);

  // clamp to valid region
  cx = std::max(0, std::min(cx, image_width - 1));
  cy = std::max(0, std::min(cy, image_height - 1));
  cw = std::max(1, std::min(cw, image_width - cx));
  ch = std::max(1, std::min(ch, image_height - cy));
  // 2) compute forward scales & pads (crop → dst)
  float sx = float(params.dst_width) / float(cw);
  float sy = float(params.dst_height) / float(ch);

  float fwd_sx, fwd_sy, pad_x = 0.0f, pad_y = 0.0f;
  if (params.keep_aspect_ratio) {
    float s = std::min(sx, sy);
    fwd_sx = fwd_sy = s;
    pad_x = (params.dst_width - cw * s) * 0.5f;
    pad_y = (params.dst_height - ch * s) * 0.5f;
  } else {
    fwd_sx = sx;
    fwd_sy = sy;
  }

  // 3) invert them so that: original = infer * inv_scale + inv_offset
  float inv_sx = 1.0f / fwd_sx;
  float inv_sy = 1.0f / fwd_sy;

  // account for both the pad *and* the crop origin
  float off_x = cx - pad_x * inv_sx;
  float off_y = cy - pad_y * inv_sy;

  return {/* scalex  */ inv_sx,
          /* scaley  */ inv_sy,
          /* offsetx */ off_x,
          /* offsety */ off_y};
}
std::shared_ptr<BaseImage> BasePreprocessor::crop(
    const std::shared_ptr<BaseImage>& image, int x, int y, int width,
    int height) {
  PreprocessParams params;
  params.dst_width = width;
  params.dst_height = height;
  params.dst_image_format = image->getImageFormat();
  params.dst_pixdata_type = image->getPixDataType();

  for (int i = 0; i < 3; i++) {
    params.mean[i] = 0;
    params.scale[i] = 1;
  }
  params.crop_x = x;
  params.crop_y = y;
  params.crop_width = width;
  params.crop_height = height;
  params.keep_aspect_ratio = false;
  return preprocess(image, params, nullptr);
}
std::shared_ptr<BaseImage> BasePreprocessor::resize(
    const std::shared_ptr<BaseImage>& image, int newWidth, int newHeight) {
  PreprocessParams params;
  params.dst_width = newWidth;
  params.dst_height = newHeight;
  params.dst_image_format = image->getImageFormat();
  params.dst_pixdata_type = image->getPixDataType();

  for (int i = 0; i < 3; i++) {
    params.mean[i] = 0;
    params.scale[i] = 1;
  }
  params.crop_x = 0;
  params.crop_y = 0;
  params.crop_width = 0;
  params.crop_height = 0;
  params.keep_aspect_ratio = false;
  return preprocess(image, params, nullptr);
}
std::shared_ptr<BaseImage> BasePreprocessor::cropResize(
    const std::shared_ptr<BaseImage>& image, int x, int y, int width,
    int height, int newWidth, int newHeight, ImageFormat dst_image_format) {
  PreprocessParams params;
  params.dst_width = newWidth;
  params.dst_height = newHeight;
  if (dst_image_format == ImageFormat::UNKOWN) {
    params.dst_image_format = image->getImageFormat();
  } else {
    params.dst_image_format = dst_image_format;
  }
  params.dst_pixdata_type = image->getPixDataType();

  for (int i = 0; i < 3; i++) {
    params.mean[i] = 0;
    params.scale[i] = 1;
  }
  params.crop_x = x;
  params.crop_y = y;
  params.crop_width = width;
  params.crop_height = height;
  params.keep_aspect_ratio = false;
  return preprocess(image, params, nullptr);
}
std::shared_ptr<BasePreprocessor> PreprocessorFactory::createPreprocessor(
    InferencePlatform platform) {
  if (platform == InferencePlatform::UNKOWN ||
      platform == InferencePlatform::AUTOMATIC) {
    platform = CommonUtils::getPlatform();
  }
  LOGI("PreprocessorFactory createPreprocessor,platform:%d\n", platform);
  switch (platform) {
    case InferencePlatform::CVITEK:
    case InferencePlatform::CV186X:
    case InferencePlatform::CV184X:
#if not defined(__BM168X__) && not defined(__CMODEL_CV181X__) && \
    not defined(__CMODEL_CV184X__)
      return std::make_shared<VpssPreprocessor>();
      // return std::make_shared<OpenCVPreprocessor>();
#else
      return nullptr;
#endif
    case InferencePlatform::BM168X:
    case InferencePlatform::CMODEL_CV181X:
    case InferencePlatform::CMODEL_CV184X:

      return std::make_shared<OpenCVPreprocessor>();

    default:
      return nullptr;
  }
  return nullptr;
}