#include "encoder/rtsp/rtsp.hpp"
#include "utils/tdl_log.hpp"

#define FRAME_MILLISEC 20000
#define RTSP_PORT 554
#define MaxPicWidth 2560
#define MaxPicHeight 1440
#define BufSize 1024 * 1024
#define Profile 0
#define IPQpDelta 2
#define Gop 25
#define StatTime 2
#define DstFrameRate 25
#define SrcFrameRate 25
#define BitRate 1024

void RTSP::onRTSPConnect(const char *ip, void *arg) {
  LOGI("RTSP client connected from: %s\n", ip);
}

void RTSP::onRTSPDisconnect(const char *ip, void *arg) {
  LOGI("RTSP client connected from: %s\n", ip);
}

int32_t RTSP::initVENC() {
  VENC_STREAM_S stream;
  VENC_PACK_S venc_pack;
  int32_t chn = context_.chn;
  stream.pstPack = &venc_pack;
  VENC_CHN_ATTR_S venc_chn_attr;
  memset(&stream, 0, sizeof(VENC_STREAM_S));
  memset(&venc_pack, 0, sizeof(VENC_PACK_S));
  memset(&venc_chn_attr, 0, sizeof(VENC_CHN_ATTR_S));

  // 设置编码器属性
  venc_chn_attr.stVencAttr.u32PicWidth = context_.frame_width;
  venc_chn_attr.stVencAttr.u32PicHeight = context_.frame_height;
  venc_chn_attr.stVencAttr.u32MaxPicWidth = MaxPicWidth;
  venc_chn_attr.stVencAttr.u32MaxPicHeight = MaxPicHeight;
  venc_chn_attr.stVencAttr.u32BufSize = BufSize;
  venc_chn_attr.stVencAttr.u32Profile = Profile;
  venc_chn_attr.stVencAttr.enType = context_.pay_load_type;
  venc_chn_attr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
  venc_chn_attr.stGopAttr.stNormalP.s32IPQpDelta = IPQpDelta;

  if (venc_chn_attr.stVencAttr.enType == PT_H264) {
    venc_chn_attr.stVencAttr.stAttrH264e.bSingleLumaBuf = CVI_FALSE;
    venc_chn_attr.stVencAttr.stAttrH264e.bRcnRefShareBuf = CVI_FALSE;
    venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
    venc_chn_attr.stRcAttr.stH264Cbr.u32Gop = Gop;
    venc_chn_attr.stRcAttr.stH264Cbr.u32StatTime = StatTime;
    venc_chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRate = DstFrameRate;
    venc_chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRate = SrcFrameRate;
    venc_chn_attr.stRcAttr.stH264Cbr.u32BitRate = BitRate;
    venc_chn_attr.stRcAttr.stH264Cbr.bVariFpsEn = CVI_FALSE;
  } else if (venc_chn_attr.stVencAttr.enType == PT_H265) {
    venc_chn_attr.stVencAttr.stAttrH265e.bRcnRefShareBuf = CVI_FALSE;
    venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H265CBR;
    venc_chn_attr.stRcAttr.stH265Cbr.u32Gop = Gop;
    venc_chn_attr.stRcAttr.stH265Cbr.u32StatTime = StatTime;
    venc_chn_attr.stRcAttr.stH265Cbr.u32SrcFrameRate = SrcFrameRate;
    venc_chn_attr.stRcAttr.stH265Cbr.fr32DstFrameRate = DstFrameRate;
    venc_chn_attr.stRcAttr.stH265Cbr.u32BitRate = BitRate;
    venc_chn_attr.stRcAttr.stH265Cbr.bVariFpsEn = CVI_FALSE;
  } else {
    return -1;
  }

  int ret = CVI_VENC_CreateChn(chn, &venc_chn_attr);
  if (ret != 0) {
    LOGE("Failed to create VENC channel");
    return ret;
  }

  VENC_CHN_PARAM_S venc_chn_param;
  memset(&venc_chn_param, 0, sizeof(VENC_CHN_PARAM_S));
  ret = CVI_VENC_GetChnParam(chn, &venc_chn_param);
  if (ret != 0) {
    LOGE("Failed to get VENC channel parameter");
    return ret;
  }

  ret = CVI_VENC_SetChnParam(chn, &venc_chn_param);
  if (ret != 0) {
    LOGE("Failed to set VENC channel parameter");
    return ret;
  }

  VENC_RECV_PIC_PARAM_S venc_recv_pic_param;
  venc_recv_pic_param.s32RecvPicNum = -1;
  ret = CVI_VENC_StartRecvFrame(chn, &venc_recv_pic_param);
  if (ret != 0) {
    LOGE("Failed to start receiving frames");
    return ret;
  }

  return 0;
}

int32_t RTSP::destroyVENC() {
  int ret = CVI_VENC_StopRecvFrame(context_.chn);
  if (ret != 0) {
    return ret;
  }
  ret = CVI_VENC_DestroyChn(context_.chn);
  if (ret != 0) {
    return ret;
  }
  return 0;
}

int32_t RTSP::initRTSP() {
  CVI_RTSP_CONFIG rtsp_config = {0};
  rtsp_config.port = RTSP_PORT;

  int32_t ret = CVI_RTSP_Create(&context_.pstRtspContext, &rtsp_config);
  if (ret != 0) {
    LOGE("Failed to create RTSP session");
    return ret;
  }

  CVI_RTSP_SESSION_ATTR attr = {0};
  if (context_.pay_load_type == PT_H264) {
    attr.video.codec = RTSP_VIDEO_H264;
    snprintf(attr.name, sizeof(attr.name), "h264");
  } else if (context_.pay_load_type == PT_H265) {
    attr.video.codec = RTSP_VIDEO_H265;
    snprintf(attr.name, sizeof(attr.name), "h265");
  } else {
    return -1;
  }

  ret = CVI_RTSP_CreateSession(context_.pstRtspContext, &attr,
                               &context_.pstSession);
  if (ret != 0) {
    LOGE("Failed to create RTSP session");
    return ret;
  }

  CVI_RTSP_STATE_LISTENER listener;
  listener.onConnect = onRTSPConnect;
  listener.argConn = context_.pstRtspContext;
  listener.onDisconnect = onRTSPDisconnect;
  CVI_RTSP_SetListener(context_.pstRtspContext, &listener);

  ret = CVI_RTSP_Start(context_.pstRtspContext);
  if (ret != 0) {
    LOGE("Failed to start RTSP");
    return ret;
  }

  return 0;
}

int32_t RTSP::destroyRTSP() {
  int32_t ret = CVI_RTSP_Stop(context_.pstRtspContext);
  if (ret != 0) {
    LOGE("Failed to destroy RTSP session");
    return ret;
  }

  ret = CVI_RTSP_DestroySession(context_.pstRtspContext, context_.pstSession);
  if (ret != 0) {
    LOGE("Failed to destroy RTSP session");
    return ret;
  }
  return 0;
}

RTSP::RTSP(int32_t chn, PAYLOAD_TYPE_E pay_load_type, int32_t frame_width,
           int32_t frame_height) {
  // 初始化RTSP上下文
  context_.chn = chn;
  context_.pay_load_type = pay_load_type;
  context_.frame_width = frame_width;
  context_.frame_height = frame_height;
  context_.pstRtspContext = nullptr;
  context_.pstSession = nullptr;

  // 初始化VENC和RTSP
  if (initVENC() != 0) {
    LOGE("Failed to initialize VENC");
  }

  if (initRTSP() != 0) {
    destroyVENC();
    LOGE("Failed to initialize RTSP");
  }
}

RTSP::~RTSP() {
  destroyRTSP();
  destroyVENC();
}

int32_t RTSP::sendFrame(VIDEO_FRAME_INFO_S *frame) {
  int32_t ret = 0;
  VENC_STREAM_S stream;
  VENC_CHN_STATUS_S venc_chn_status;
  VENC_CHN chn = context_.chn;
  VENC_PACK_S *venc_pack_ptr;
  CVI_RTSP_DATA rtsp_data = {0};

  ret = CVI_VENC_SendFrame(chn, frame, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("Failed to send frame");
    return ret;
  }

  ret = CVI_VENC_QueryStatus(chn, &venc_chn_status);
  if (ret != 0) {
    LOGE("Failed to query VENC status");
    return ret;
  }

  stream.pstPack =
      (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S) * venc_chn_status.u32CurPacks);
  ret = CVI_VENC_GetStream(chn, &stream, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("Failed to get VENC stream");
    free(stream.pstPack);
    return ret;
  }

  rtsp_data.blockCnt = stream.u32PackCount;
  for (unsigned int i = 0; i < stream.u32PackCount; i++) {
    venc_pack_ptr = &stream.pstPack[i];
    rtsp_data.dataPtr[i] = venc_pack_ptr->pu8Addr + venc_pack_ptr->u32Offset;
    rtsp_data.dataLen[i] = venc_pack_ptr->u32Len - venc_pack_ptr->u32Offset;
  }

  ret = CVI_RTSP_WriteFrame(context_.pstRtspContext, context_.pstSession->video,
                            &rtsp_data);

  CVI_VENC_ReleaseStream(chn, &stream);
  free(stream.pstPack);
  return ret;
}