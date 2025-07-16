#ifndef _RTSP_UTILS_H_
#define _RTSP_UTILS_H_

#include <cvi_comm_video.h>
#include <stdio.h>
#include <stdlib.h>
#include "tdl_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t chn;
  PAYLOAD_TYPE_E pay_load_type;
  int32_t frame_width;
  int32_t frame_height;
} TDLRTSPContext;

int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S *frame,
                          TDLRTSPContext *rtsp_context);

#ifdef __cplusplus
}
#endif

#endif  // _RTSP_UTILS_H_