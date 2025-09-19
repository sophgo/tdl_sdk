#ifndef _RTSP_HPP_
#define _RTSP_HPP_

#include <cvi_type.h>
#include <cvi_venc.h>
#include <rtsp.h>
#include <cstring>
#include <stdexcept>

struct RTSPContext {
  int32_t chn;
  PAYLOAD_TYPE_E pay_load_type;
  int32_t frame_width;
  int32_t frame_height;
  CVI_RTSP_CTX *pstRtspContext;
  CVI_RTSP_SESSION *pstSession;
};

class RTSP {
 public:
  RTSP(int32_t chn = 0, PAYLOAD_TYPE_E pay_load_type = PT_H264,
       int32_t frame_width = 1920, int32_t frame_height = 1080);
  ~RTSP();

  int32_t sendFrame(VIDEO_FRAME_INFO_S *frame);

 private:
  static void onRTSPConnect(const char *ip, void *arg);
  static void onRTSPDisconnect(const char *ip, void *arg);

  int32_t initVENC();
  int32_t destroyVENC();
  int32_t initRTSP();
  int32_t destroyRTSP();

 private:
  RTSPContext context_;
};

#endif  // _RTSP_HPP_
