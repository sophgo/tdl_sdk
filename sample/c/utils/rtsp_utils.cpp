#include "rtsp_utils.h"
#include <memory>
#include "encoder/rtsp/rtsp.hpp"
#include "utils/tdl_log.hpp"

namespace {
// 全局存储RTSP实例
std::shared_ptr<RTSP> g_rtsp_instance;
}  // namespace

extern "C" {

int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S* frame,
                          TDLRTSPContext* rtsp_context) {
  if (g_rtsp_instance == nullptr) {
    g_rtsp_instance = std::make_shared<RTSP>(
        rtsp_context->chn, rtsp_context->pay_load_type,
        rtsp_context->frame_width, rtsp_context->frame_height);
  }
  g_rtsp_instance->sendFrame(frame);
  return 0;
}

}  // extern "C"
