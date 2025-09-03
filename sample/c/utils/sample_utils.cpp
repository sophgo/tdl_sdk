#include "sample_utils.h"
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

int32_t SendFrameRTSP(VIDEO_FRAME_INFO_S *frame, RtspContext *rtsp_context) {
  if (g_rtsp_instance == nullptr) {
    g_rtsp_instance = std::make_shared<RTSP>(
        rtsp_context->chn, rtsp_context->pay_load_type,
        rtsp_context->frame_width, rtsp_context->frame_height);
  }
  g_rtsp_instance->sendFrame(frame);
  return 0;
}

void InitQueue(ImageQueue *q) {
  q->front = q->rear = q->count = 0;
  pthread_mutex_init(&q->mutex, NULL);
}

void DestroyQueue(ImageQueue *q) {
  pthread_mutex_lock(&q->mutex);
  while (q->count > 0) {
    TDL_DestroyImage(q->queue[q->front]);
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;
  }
  pthread_mutex_unlock(&q->mutex);
  pthread_mutex_destroy(&q->mutex);
}

int Image_Enqueue(ImageQueue *q, TDLImage img) {
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

TDLImage Image_Dequeue(ImageQueue *q) {
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
static ImageFormat ConvertPixelFormat(ImageFormatE image_fmt) {
  switch (image_fmt) {
    case IMAGE_GRAY:
      return ImageFormat::GRAY;
    case IMAGE_RGB_PLANAR:
      return ImageFormat::RGB_PLANAR;
    case IMAGE_RGB_PACKED:
      return ImageFormat::RGB_PACKED;
    case IMAGE_BGR_PLANAR:
      return ImageFormat::BGR_PLANAR;
    case IMAGE_BGR_PACKED:
      return ImageFormat::BGR_PACKED;
    case IMAGE_YUV420SP_UV:
      return ImageFormat::YUV420SP_UV;
    case IMAGE_YUV420SP_VU:
      return ImageFormat::YUV420SP_VU;
    case IMAGE_YUV420P_UV:
      return ImageFormat::YUV420P_UV;
    case IMAGE_YUV420P_VU:
      return ImageFormat::YUV420P_VU;
    case IMAGE_YUV422P_UV:
      return ImageFormat::YUV422P_UV;
    case IMAGE_YUV422P_VU:
      return ImageFormat::YUV422P_VU;
    case IMAGE_YUV422SP_UV:
      return ImageFormat::YUV422SP_UV;
    case IMAGE_YUV422SP_VU:
      return ImageFormat::YUV422SP_VU;
    default:
      return ImageFormat::UNKOWN;
  }
}

int32_t InitCamera(TDLHandle handle, int w, int h, ImageFormatE image_fmt,
                   int vb_buffer_num) {
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

  context->video_decoder->initialize(w, h, ConvertPixelFormat(image_fmt),
                                     vb_buffer_num);

  return 0;
}

TDLImage GetCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;

  TDLImageContext *image_context = new TDLImageContext();

  context->video_decoder->read(image_context->image, chn);

  return (TDLImage)image_context;
}

int32_t ReleaseCameraFrame(TDLHandle handle, int chn) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder->release(chn) != 0) {
    LOGE("release camera frame failed\n");
    return -1;
  }
  return 0;
}

int32_t DestoryCamera(TDLHandle handle) {
  TDLContext *context = (TDLContext *)handle;
  if (context->video_decoder != nullptr) {
    context->video_decoder.reset();
    context->video_decoder = nullptr;
  }
  return 0;
}

#endif

}  // extern "C"
