#ifndef _RTSP_UTILS_H_
#define _RTSP_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <cvi_type.h>
#include <cvi_venc.h>
#include <rtsp.h>
#include <stdio.h>
#include <stdlib.h>
#include "tdl_sdk.h"

typedef struct {
  int32_t venc_chn;
  PAYLOAD_TYPE_E pay_load_type;
  CVI_RTSP_CTX *pstRtspContext;
  CVI_RTSP_SESSION *pstSession;
} TDLRTSPContext;

typedef struct {
  int32_t venc_chn;
  PAYLOAD_TYPE_E pay_load_type;
  int32_t frame_width;
  int32_t frame_height;
} TDLVENCContext;

int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S *frame,
                          TDLRTSPContext *rtsp_context_ptr);

int32_t TDL_InitVENC(TDLVENCContext *venc_context_ptr);

int32_t TDL_DestroyVENC(TDLVENCContext *venc_context_ptr);

int32_t TDL_InitRTSP(TDLRTSPContext *rtsp_context_ptr);

int32_t TDL_DestroyRTSP(TDLRTSPContext *rtsp_context_ptr);

int32_t TDL_WrapImage(TDLImage image, VIDEO_FRAME_INFO_S *frame);

#ifdef __cplusplus
}
#endif

#endif  // _RTSP_UTILS_H_