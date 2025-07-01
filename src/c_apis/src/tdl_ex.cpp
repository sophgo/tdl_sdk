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
int32_t TDL_InitCamera(TDLHandle handle) {
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
