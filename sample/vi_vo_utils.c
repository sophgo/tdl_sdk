
#include "vi_vo_utils.h"
#include <cvi_venc.h>
#include <stdlib.h>
#include "core/utils/vpss_helper.h"

CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *pstViConfig, CVI_U32 *devNum) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  SAMPLE_INI_CFG_S stIniCfg = {
      .enSource = VI_PIPE_FRAME_SOURCE_DEV,
      .devNum = 1,
      .enSnsType = SONY_IMX327_MIPI_2M_30FPS_12BIT,
      .enWDRMode = WDR_MODE_NONE,
      .s32BusId = 3,
      .s32SnsI2cAddr = -1,
      .MipiDev = 0xFF,
      .u8UseDualSns = 0,
      .enSns2Type = SONY_IMX327_SLAVE_MIPI_2M_30FPS_12BIT,
      .s32Sns2BusId = 0,
      .s32Sns2I2cAddr = -1,
      .Sns2MipiDev = 0xFF,
  };

  if (!SAMPLE_COMM_VI_ParseIni(&stIniCfg)) {
    syslog(LOG_ERR | LOG_LOCAL7, "Init pasre failed.\n");
    return CVI_FAILURE;
  }
  *devNum = stIniCfg.devNum;

  DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
  PIXEL_FORMAT_E enPixFormat = VI_PIXEL_FORMAT;
  VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
  VI_VPSS_MODE_E enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;

  VI_CHN viChn = 0;
  CVI_S32 s32WorkSnsId = 0;
  PIC_SIZE_E enPicSize;
  SIZE_S stSize;

  SAMPLE_COMM_VI_GetSensorInfo(pstViConfig);
  for (; s32WorkSnsId < stIniCfg.devNum; s32WorkSnsId++) {
    pstViConfig->s32WorkingViNum = 1 + s32WorkSnsId;
    pstViConfig->as32WorkingViId[s32WorkSnsId] = s32WorkSnsId;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType =
        (s32WorkSnsId == 0) ? stIniCfg.enSnsType : stIniCfg.enSns2Type;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.MipiDev =
        (s32WorkSnsId == 0) ? stIniCfg.MipiDev : stIniCfg.Sns2MipiDev;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.s32BusId =
        (s32WorkSnsId == 0) ? stIniCfg.s32BusId : stIniCfg.s32Sns2BusId;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[0] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[0] : stIniCfg.as16Sns2LaneId[0];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[1] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[1] : stIniCfg.as16Sns2LaneId[1];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[2] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[2] : stIniCfg.as16Sns2LaneId[2];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[3] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[3] : stIniCfg.as16Sns2LaneId[3];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[4] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[4] : stIniCfg.as16Sns2LaneId[4];

    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[0] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[0] : stIniCfg.as8Sns2PNSwap[0];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[1] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[1] : stIniCfg.as8Sns2PNSwap[1];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[2] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[2] : stIniCfg.as8Sns2PNSwap[2];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[3] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[3] : stIniCfg.as8Sns2PNSwap[3];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[4] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[4] : stIniCfg.as8Sns2PNSwap[4];

    pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.ViDev = 0;
    pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.enWDRMode =
        (s32WorkSnsId == 0) ? stIniCfg.enWDRMode : stIniCfg.enSns2WDRMode;

    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0] = s32WorkSnsId;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1] = -1;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2] = -1;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3] = -1;

    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.ViChn = viChn;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enPixFormat = enPixFormat;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange = enDynamicRange;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat = enVideoFormat;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode = enCompressMode;
  }

  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType, &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = MMF_INIT_HELPER2(stSize.u32Width, stSize.u32Height, enPixFormat, 10, stSize.u32Width,
                            stSize.u32Height, enPixFormat, 10);
  if (s32Ret != CVI_SUCCESS) {
    printf("sys init failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_PLAT_VI_INIT(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi init failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }
  return s32Ret;
}

static CVI_S32 InitVO(const CVI_U32 width, const CVI_U32 height, SAMPLE_VO_CONFIG_S *stVoConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  s32Ret = SAMPLE_COMM_VO_GetDefConfig(stVoConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
    return s32Ret;
  }
  RECT_S dispRect = {0, 0, height, width};
  SIZE_S imgSize = {height, width};

  stVoConfig->VoDev = 0;
  stVoConfig->stVoPubAttr.enIntfType = VO_INTF_MIPI;
  stVoConfig->stVoPubAttr.enIntfSync = VO_OUTPUT_720x1280_60;
  stVoConfig->stDispRect = dispRect;
  stVoConfig->stImageSize = imgSize;
  stVoConfig->enPixFormat = VI_PIXEL_FORMAT;
  stVoConfig->enVoMode = VO_MODE_1MUX;

  s32Ret = SAMPLE_COMM_VO_StartVO(stVoConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", s32Ret);
  }
  CVI_VO_SetChnRotation(0, 0, ROTATION_270);
  printf("SAMPLE_COMM_VO_StartVO done\n");
  return s32Ret;
}

CVI_S32 InitVPSS(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                 const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                 const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

  abChnEnable[vpssChn] = CVI_TRUE;
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChn], voWidth, voHeight, PIXEL_FORMAT_RGB_888, true);

  if (isVOOpened) {
    abChnEnable[vpssChnVO] = CVI_TRUE;
    VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChnVO], voWidth, voHeight,
                            PIXEL_FORMAT_YUV_PLANAR_420, true);
  }

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, grpWidth, grpHeight, VI_PIXEL_FORMAT);

  /*start vpss*/
  s32Ret = SAMPLE_COMM_VPSS_Init(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VPSS_Start(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_Bind_VPSS(viPipe, vpssChn, vpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}

static void _initInputCfg(chnInputCfg *ipIc) {
  ipIc->rcMode = -1;
  ipIc->iqp = -1;
  ipIc->pqp = -1;
  ipIc->gop = -1;
  ipIc->bitrate = -1;
  ipIc->firstFrmstartQp = -1;
  ipIc->num_frames = -1;
  ipIc->framerate = 30;
  ipIc->maxQp = -1;
  ipIc->minQp = -1;
  ipIc->maxIqp = -1;
  ipIc->minIqp = -1;
}

static void rtsp_connect(const char *ip, void *arg) { printf("connect: %s\n", ip); }

static void rtsp_disconnect(const char *ip, void *arg) { printf("disconnect: %s\n", ip); }

static PIC_SIZE_E get_output_size(CVI_S32 width, CVI_S32 height) {
  if (width == 1280 && height == 720) {
    return PIC_720P;
  } else if (width == 1920 && height == 1080) {
    return PIC_1080P;
  } else {
    return PIC_BUTT;
  }
}

static CVI_S32 InitRTSP(VencCodec codec, CVI_S32 frameWidth, CVI_S32 frameHeight,
                        OutputContext *context) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VENC_CHN VencChn[] = {0};
  PAYLOAD_TYPE_E enPayLoad = codec == CODEC_H264 ? PT_H264 : PT_H265;
  PIC_SIZE_E enSize = get_output_size(frameWidth, frameHeight);
  if (enSize == PIC_BUTT) {
    printf("Unsupported resolution: (%#x, %#x)", frameWidth, frameHeight);
    return CVI_FAILURE;
  }

  VENC_GOP_MODE_E enGopMode = VENC_GOPMODE_NORMALP;
  VENC_GOP_ATTR_S stGopAttr;
  SAMPLE_RC_E enRcMode;
  CVI_U32 u32Profile = 0;

  _initInputCfg(&context->input_cfg);
  strcpy(context->input_cfg.codec, codec == CODEC_H264 ? "264" : "265");

  context->input_cfg.rcMode = 0;  // cbr
  context->input_cfg.iqp = 38;
  context->input_cfg.pqp = 38;
  context->input_cfg.gop = 50;
  context->input_cfg.bitrate = 10240;  // if fps = 20
  context->input_cfg.firstFrmstartQp = 34;
  context->input_cfg.num_frames = -1;
  context->input_cfg.framerate = 25;
  context->input_cfg.srcFramerate = 25;
  context->input_cfg.maxQp = 42;
  context->input_cfg.minQp = 26;
  context->input_cfg.maxIqp = 42;
  context->input_cfg.minIqp = 26;

  enRcMode = (SAMPLE_RC_E)context->input_cfg.rcMode;

  s32Ret = SAMPLE_COMM_VENC_GetGopAttr(enGopMode, &stGopAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("[Err]Venc Get GopAttr for %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  s32Ret = SAMPLE_COMM_VENC_Start(&context->input_cfg, VencChn[0], enPayLoad, enSize, enRcMode,
                                  u32Profile, CVI_FALSE, &stGopAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("[Err]Venc Start failed for %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  CVI_RTSP_CONFIG config = {0};
  config.port = 554;

  context->rtsp_context = NULL;
  if (0 > CVI_RTSP_Create(&context->rtsp_context, &config)) {
    printf("fail to create rtsp contex\n");
    return CVI_FAILURE;
  }

  context->session = NULL;
  CVI_RTSP_SESSION_ATTR attr = {0};
  attr.video.codec = codec == CODEC_H264 ? RTSP_VIDEO_H264 : RTSP_VIDEO_H265;

  snprintf(attr.name, sizeof(attr.name), "%s", codec == CODEC_H264 ? "h264" : "h265");

  CVI_RTSP_CreateSession(context->rtsp_context, &attr, &context->session);

  // set listener
  context->listener.onConnect = rtsp_connect;
  context->listener.argConn = context->rtsp_context;
  context->listener.onDisconnect = rtsp_disconnect;

  CVI_RTSP_SetListener(context->rtsp_context, &context->listener);

  if (0 > CVI_RTSP_Start(context->rtsp_context)) {
    printf("fail to start\n");
    return CVI_FAILURE;
  }
  printf("init done\n");
  return s32Ret;
}

CVI_S32 InitOutput(OutputType outputType, CVI_S32 frameWidth, CVI_S32 frameHeight,
                   OutputContext *context) {
  context->type = outputType;
  CVI_S32 s32Ret = CVI_SUCCESS;
  switch (outputType) {
    case OUTPUT_TYPE_PANEL: {
      printf("Init panel\n");
      context->voChn = 0;
      context->voLayer = 0;
      SAMPLE_VO_CONFIG_S voConfig;
      s32Ret = InitVO(frameWidth, frameHeight, &voConfig);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
        return s32Ret;
      }
      CVI_VO_HideChn(context->voLayer, context->voChn);
      return CVI_SUCCESS;
    }
    case OUTPUT_TYPE_RTSP: {
      printf("Init rtsp\n");
      return InitRTSP(CODEC_H264, frameWidth, frameHeight, context);
    }
    default:
      printf("Unsupported output typed: %x\n", outputType);
      return CVI_FAILURE;
  };

  return CVI_SUCCESS;
}

static CVI_S32 panel_send_frame(VIDEO_FRAME_INFO_S *stVencFrame, OutputContext *context) {
  CVI_S32 s32Ret = CVI_VO_SendFrame(context->voLayer, context->voChn, stVencFrame, -1);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VO_SendFrame failed with %#x\n", s32Ret);
  }
  CVI_VO_ShowChn(context->voLayer, context->voChn);
  return s32Ret;
}

static CVI_S32 rtsp_send_frame(VIDEO_FRAME_INFO_S *stVencFrame, OutputContext *context) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_RTSP_SESSION *session = context->session;

  CVI_S32 s32SetFrameMilliSec = 20000;
  VENC_STREAM_S stStream;
  VENC_CHN_ATTR_S stVencChnAttr;
  VENC_CHN_STATUS_S stStat;
  VENC_CHN VencChn = 0;
  s32Ret = CVI_VENC_SendFrame(VencChn, stVencFrame, s32SetFrameMilliSec);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VENC_SendFrame failed! %d\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_VENC_GetChnAttr(VencChn, &stVencChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VENC_GetChnAttr, VencChn[%d], s32Ret = %d\n", VencChn, s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_VENC_QueryStatus(VencChn, &stStat);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VENC_QueryStatus failed with %#x!\n", s32Ret);
    return s32Ret;
  }
  if (!stStat.u32CurPacks) {
    printf("NOTE: Current frame is NULL!\n");
    return s32Ret;
  }

  stStream.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S) * stStat.u32CurPacks);
  if (stStream.pstPack == NULL) {
    printf("malloc memory failed!\n");
    return s32Ret;
  }

  s32Ret = CVI_VENC_GetStream(VencChn, &stStream, -1);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VENC_GetStream failed with %#x!\n", s32Ret);
    free(stStream.pstPack);
    stStream.pstPack = NULL;
    return s32Ret;
  }

  VENC_PACK_S *ppack;
  CVI_RTSP_DATA data = {0};
  memset(&data, 0, sizeof(CVI_RTSP_DATA));

  data.blockCnt = stStream.u32PackCount;
  for (unsigned int i = 0; i < stStream.u32PackCount; i++) {
    ppack = &stStream.pstPack[i];
    data.dataPtr[i] = ppack->pu8Addr + ppack->u32Offset;
    data.dataLen[i] = ppack->u32Len - ppack->u32Offset;
  }

  CVI_RTSP_WriteFrame(context->rtsp_context, session->video, &data);

  s32Ret = CVI_VENC_ReleaseStream(VencChn, &stStream);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VENC_ReleaseStream, s32Ret = %d\n", s32Ret);
    free(stStream.pstPack);
    stStream.pstPack = NULL;
    return s32Ret;
  }

  free(stStream.pstPack);
  stStream.pstPack = NULL;
  return s32Ret;
}

CVI_S32 SendOutputFrame(VIDEO_FRAME_INFO_S *stVencFrame, OutputContext *context) {
  if (context->type == OUTPUT_TYPE_PANEL) {
    return panel_send_frame(stVencFrame, context);
  } else if (context->type == OUTPUT_TYPE_RTSP) {
    return rtsp_send_frame(stVencFrame, context);
  } else {
    printf("Failed to send output frame: Wrong output type(%x)\n", context->type);
    return CVI_FAILURE;
  }
}

CVI_S32 DestoryOutput(OutputContext *context) {
  if (context->type == OUTPUT_TYPE_RTSP) {
    CVI_RTSP_Stop(context->rtsp_context);
    CVI_RTSP_DestroySession(context->rtsp_context, context->session);
    CVI_RTSP_Destroy(&context->rtsp_context);
    SAMPLE_COMM_VENC_Stop(0);
  }
  return CVI_SUCCESS;
}