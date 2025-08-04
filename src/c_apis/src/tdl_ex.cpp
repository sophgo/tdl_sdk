#include "tdl_sdk.h"

#include <opencv2/opencv.hpp>
#include "app/app_data_types.hpp"
#include "common/common_types.hpp"
#include "tdl_type_internal.hpp"
#include "tdl_utils.h"
#include "utils/common_utils.hpp"
#include "utils/tdl_log.hpp"
#include "video_decoder/video_decoder_type.hpp"

#if !defined(__BM168X__) && !defined(__CMODEL_CV181X__)
static ImageFormat TDL_ConvertPixelFormat(TDLImageFormatE image_fmt) {
  switch (image_fmt) {
    case TDL_IMAGE_GRAY:
      return ImageFormat::GRAY;
    case TDL_IMAGE_RGB_PLANAR:
      return ImageFormat::RGB_PLANAR;
    case TDL_IMAGE_RGB_PACKED:
      return ImageFormat::RGB_PACKED;
    case TDL_IMAGE_BGR_PLANAR:
      return ImageFormat::BGR_PLANAR;
    case TDL_IMAGE_BGR_PACKED:
      return ImageFormat::BGR_PACKED;
    case TDL_IMAGE_YUV420SP_UV:
      return ImageFormat::YUV420SP_UV;
    case TDL_IMAGE_YUV420SP_VU:
      return ImageFormat::YUV420SP_VU;
    case TDL_IMAGE_YUV420P_UV:
      return ImageFormat::YUV420P_UV;
    case TDL_IMAGE_YUV420P_VU:
      return ImageFormat::YUV420P_VU;
    case TDL_IMAGE_YUV422P_UV:
      return ImageFormat::YUV422P_UV;
    case TDL_IMAGE_YUV422P_VU:
      return ImageFormat::YUV422P_VU;
    case TDL_IMAGE_YUV422SP_UV:
      return ImageFormat::YUV422SP_UV;
    case TDL_IMAGE_YUV422SP_VU:
      return ImageFormat::YUV422SP_VU;
    default:
      return ImageFormat::UNKOWN;
  }
}

int32_t TDL_InitCamera(TDLHandle handle, int w, int h,
                       TDLImageFormatE image_fmt, int vb_buffer_num) {
  TDLContext *context = (TDLContext *)handle;
  if (context == nullptr) {
    return -1;
  }

  context->video_decoder =
      VideoDecoderFactory::createVideoDecoder(VideoDecoderType::VI);
  if (context->video_decoder == nullptr) {
    LOGE("create video decoder failed\n");
    return -1;
  }

  context->video_decoder->initialize(w, h, TDL_ConvertPixelFormat(image_fmt),
                                     vb_buffer_num);

  return 0;
}

TDLImage TDL_GetCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;

  TDLImageContext *image_context = new TDLImageContext();

  context->video_decoder->read(image_context->image, chn);

  return (TDLImage)image_context;
}

int32_t TDL_ReleaseCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder->release(chn) != 0) {
    LOGE("release camera frame failed\n");
    return -1;
  }
  return 0;
}

int32_t TDL_DestoryCamera(TDLHandle handle) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder != nullptr) {
    context->video_decoder.reset();
    context->video_decoder = nullptr;
  }
  return 0;
}

#endif
