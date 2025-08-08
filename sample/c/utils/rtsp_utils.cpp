#include "rtsp_utils.h"
#include <memory>
#include "encoder/rtsp/rtsp.hpp"
#include "tdl_type_internal.hpp"
#include "utils/tdl_log.hpp"
#include "video_decoder/video_decoder_type.hpp"

namespace {
// 全局存储RTSP实例
std::shared_ptr<RTSP> g_rtsp_instance;
}  // namespace

extern "C" {

int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S *frame,
                          TDLRTSPContext *rtsp_context) {
  if (g_rtsp_instance == nullptr) {
    g_rtsp_instance = std::make_shared<RTSP>(
        rtsp_context->chn, rtsp_context->pay_load_type,
        rtsp_context->frame_width, rtsp_context->frame_height);
  }
  g_rtsp_instance->sendFrame(frame);
  return 0;
}

void TDL_InitQueue(TDLImageQueue *q) {
  q->front = q->rear = q->count = 0;
  pthread_mutex_init(&q->mutex, NULL);
}

void TDL_DestroyQueue(TDLImageQueue *q) {
  pthread_mutex_lock(&q->mutex);
  while (q->count > 0) {
    TDL_DestroyImage(q->queue[q->front]);
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;
  }
  pthread_mutex_unlock(&q->mutex);
  pthread_mutex_destroy(&q->mutex);
}

int TDL_Image_Enqueue(TDLImageQueue *q, TDLImage img) {
  int ret = 0;
  pthread_mutex_lock(&q->mutex);
  if (q->count == QUEUE_SIZE) {
    ret = -1;
  } else {
    q->queue[q->rear] = img;
    q->rear = (q->rear + 1) % QUEUE_SIZE;
    q->count++;
    ret = 0;
  }
  pthread_mutex_unlock(&q->mutex);
  return ret;
}

TDLImage TDL_Image_Dequeue(TDLImageQueue *q) {
  TDLImage img = NULL;
  pthread_mutex_lock(&q->mutex);
  if (q->count == 0) {
    img = NULL;
  } else {
    img = q->queue[q->front];
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;
  }
  pthread_mutex_unlock(&q->mutex);
  return img;
}

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

}  // extern "C"
