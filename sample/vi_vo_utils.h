#ifndef VI_VO_UTILS_H_
#define VI_VO_UTILS_H_

#include <cvi_sys.h>
#include <cvi_vi.h>
#include <pthread.h>
#include <rtsp.h>

#include "sample_comm.h"

typedef enum {
  CODEC_H264,
  CODEC_H265,
} VencCodec;

typedef enum { OUTPUT_TYPE_PANEL, OUTPUT_TYPE_RTSP } OutputType;

typedef struct {
  OutputType type;
  union {
    struct {
      CVI_S32 voLayer;
      CVI_S32 voChn;
    };

    struct {
      CVI_RTSP_STATE_LISTENER listener;
      CVI_RTSP_CTX *rtsp_context;
      CVI_RTSP_SESSION *session;
      chnInputCfg input_cfg;
    };
  };
} OutputContext;
CVI_RTSP_CTX a;

CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *pstViConfig, CVI_U32 *devNum);

CVI_S32 InitVPSS_RGB(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                     const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                     const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened);

CVI_S32 InitVPSS(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                 const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                 const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened,
                 PIXEL_FORMAT_E format);

CVI_S32 InitOutput(OutputType outputType, CVI_S32 frameWidth, CVI_S32 frameHeight,
                   OutputContext *context);
CVI_S32 SendOutputFrame(VIDEO_FRAME_INFO_S *stVencFrame, OutputContext *context);
CVI_S32 DestoryOutput(OutputContext *context);

#endif