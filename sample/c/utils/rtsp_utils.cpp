#include "rtsp_utils.h"
#include "tdl_type_internal.hpp"
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
#ifdef __cplusplus
extern "C" {
#endif

static void TDL_RTSPOnConnect(const char *ip, void *arg) {
  LOGI("RTSP client connected from: %s\n", ip);
}

static void TDL_RTSPOnDisconnect(const char *ip, void *arg) {
  LOGI("RTSP client disconnected from: %s\n", ip);
}

int32_t TDL_InitVENC(TDLVENCContext *venc_context_ptr) {
  VENC_STREAM_S stream;
  VENC_PACK_S venc_pack;
  int32_t venc_chn = venc_context_ptr->venc_chn;
  stream.pstPack = &venc_pack;
  VENC_CHN_ATTR_S venc_chn_attr;
  memset(&stream, 0, sizeof(VENC_STREAM_S));
  memset(&venc_pack, 0, sizeof(VENC_PACK_S));
  memset(&venc_chn_attr, 0, sizeof(VENC_CHN_ATTR_S));

  // 设置编码器属性
  venc_chn_attr.stVencAttr.u32PicWidth = venc_context_ptr->frame_width;
  venc_chn_attr.stVencAttr.u32PicHeight = venc_context_ptr->frame_height;
  venc_chn_attr.stVencAttr.u32MaxPicWidth = MaxPicWidth;
  venc_chn_attr.stVencAttr.u32MaxPicHeight = MaxPicHeight;
  venc_chn_attr.stVencAttr.u32BufSize = BufSize;
  venc_chn_attr.stVencAttr.u32Profile = Profile;
  venc_chn_attr.stVencAttr.enType = venc_context_ptr->pay_load_type;
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
    LOGE("Unsupported encode type: %d\n", venc_chn_attr.stVencAttr.enType);
    return -1;
  }

  int ret = CVI_VENC_CreateChn(venc_chn, &venc_chn_attr);
  if (ret != 0) {
    LOGE("Create VENC channel failed! Error: %#x\n", ret);
    return ret;
  }

  // 设置编码器参数
  VENC_CHN_PARAM_S venc_chn_param;
  memset(&venc_chn_param, 0, sizeof(VENC_CHN_PARAM_S));
  ret = CVI_VENC_GetChnParam(venc_chn, &venc_chn_param);
  if (ret != 0) {
    LOGE("CVI_VENC_GetChnParam failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_VENC_SetChnParam(venc_chn, &venc_chn_param);
  if (ret != 0) {
    LOGE("CVI_VENC_SetChnParam failed with %#x!\n", ret);
    return ret;
  }

  // 启动接收帧
  VENC_RECV_PIC_PARAM_S venc_recv_pic_param;
  venc_recv_pic_param.s32RecvPicNum = -1;  // 持续接收
  ret = CVI_VENC_StartRecvFrame(venc_chn, &venc_recv_pic_param);
  if (ret != 0) {
    LOGE("CVI_VENC_StartRecvFrame failed with %#x\n", ret);
    return ret;
  }
  return ret;
}

int32_t TDL_DestroyVENC(TDLVENCContext *venc_context_ptr) {
  int ret = 0;
  int32_t venc_chn = venc_context_ptr->venc_chn;
  ret = CVI_VENC_StopRecvFrame(venc_chn);
  if (ret != 0) {
    LOGE("CVI_VENC_StopRecvFrame failed with %#x\n", ret);
    return ret;
  }
  ret = CVI_VENC_DestroyChn(venc_chn);
  if (ret != 0) {
    LOGE("CVI_VENC_DestroyChn failed with %#x\n", ret);
    return ret;
  }
  return ret;
}

int32_t TDL_SendFrameRTSP(VIDEO_FRAME_INFO_S *frame,
                          TDLRTSPContext *rtsp_context_ptr) {
  int32_t ret = 0;
  VENC_STREAM_S stream;
  VENC_CHN_ATTR_S venc_chn_attr;
  VENC_CHN_STATUS_S venc_chn_status;
  VENC_CHN venc_chn = rtsp_context_ptr->venc_chn;
  VENC_PACK_S *venc_pack_ptr;
  CVI_RTSP_DATA rtsp_data = {0};

  ret = CVI_VENC_SendFrame(venc_chn, frame, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("CVI_VENC_SendFrame failed! %d\n", ret);
    return ret;
  }
  ret = CVI_VENC_GetChnAttr(venc_chn, &venc_chn_attr);
  if (ret != 0) {
    LOGE("CVI_VENC_GetChnAttr, venc_chn[%d], ret = %d\n", venc_chn, ret);
    return ret;
  }
  ret = CVI_VENC_QueryStatus(venc_chn, &venc_chn_status);
  if (ret != 0) {
    LOGE("CVI_VENC_QueryStatus failed with %#x!\n", ret);
    return ret;
  }
  stream.pstPack =
      (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S) * venc_chn_status.u32CurPacks);
  ret = CVI_VENC_GetStream(venc_chn, &stream, FRAME_MILLISEC);
  if (ret != 0) {
    LOGE("CVI_VENC_GetStream failed with %#x!\n", ret);
    CVI_VENC_ReleaseStream(venc_chn, &stream);
    free(stream.pstPack);
    return ret;
  }

  rtsp_data.blockCnt = stream.u32PackCount;
  for (unsigned int i = 0; i < stream.u32PackCount; i++) {
    venc_pack_ptr = &stream.pstPack[i];
    rtsp_data.dataPtr[i] = venc_pack_ptr->pu8Addr + venc_pack_ptr->u32Offset;
    rtsp_data.dataLen[i] = venc_pack_ptr->u32Len - venc_pack_ptr->u32Offset;
  }

  ret = CVI_RTSP_WriteFrame(rtsp_context_ptr->pstRtspContext,
                            rtsp_context_ptr->pstSession->video, &rtsp_data);
  if (ret != 0) {
    LOGE("CVI_RTSP_WriteFrame failed with %#x!\n", ret);
  }

  CVI_VENC_ReleaseStream(venc_chn, &stream);
  free(stream.pstPack);
  return ret;
}

int32_t TDL_InitRTSP(TDLRTSPContext *rtsp_context_ptr) {
  int32_t ret = 0;
  PAYLOAD_TYPE_E pay_load_type = rtsp_context_ptr->pay_load_type;
  CVI_RTSP_CONFIG rtsp_config = {0};
  memset(&rtsp_config, 0, sizeof(CVI_RTSP_CONFIG));
  rtsp_config.port = RTSP_PORT;

  // RTSP
  LOGI("Initialize RTSP\n");
  ret = CVI_RTSP_Create(&rtsp_context_ptr->pstRtspContext, &rtsp_config);
  if (ret != 0) {
    LOGE("fail to create rtsp context\n");
    return ret;
  }

  CVI_RTSP_SESSION_ATTR attr = {0};
  if (pay_load_type == PT_H264) {
    attr.video.codec = RTSP_VIDEO_H264;
    snprintf(attr.name, sizeof(attr.name), "h264");
  } else if (pay_load_type == PT_H265) {
    attr.video.codec = RTSP_VIDEO_H265;
    snprintf(attr.name, sizeof(attr.name), "h265");
  } else {
    LOGE("Unsupported encode type: %d\n", pay_load_type);
    return -1;
  }

  CVI_RTSP_CreateSession(rtsp_context_ptr->pstRtspContext, &attr,
                         &rtsp_context_ptr->pstSession);

  // Set listener to RTSP
  CVI_RTSP_STATE_LISTENER listener;
  listener.onConnect = TDL_RTSPOnConnect;
  listener.argConn = rtsp_context_ptr->pstRtspContext;
  listener.onDisconnect = TDL_RTSPOnDisconnect;
  CVI_RTSP_SetListener(rtsp_context_ptr->pstRtspContext, &listener);
  ret = CVI_RTSP_Start(rtsp_context_ptr->pstRtspContext);
  if (ret != 0) {
    LOGE("CVI_RTSP_Start failed with %#x!\n", ret);
    CVI_RTSP_DestroySession(rtsp_context_ptr->pstRtspContext,
                            rtsp_context_ptr->pstSession);
    CVI_RTSP_Destroy(&rtsp_context_ptr->pstRtspContext);
    return ret;
  }

  return ret;
}

int32_t TDL_DestroyRTSP(TDLRTSPContext *rtsp_context_ptr) {
  int32_t ret = 0;
  ret = CVI_RTSP_Stop(rtsp_context_ptr->pstRtspContext);
  if (ret != 0) {
    LOGE("CVI_RTSP_Stop failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_RTSP_DestroySession(rtsp_context_ptr->pstRtspContext,
                                rtsp_context_ptr->pstSession);
  if (ret != 0) {
    LOGE("CVI_RTSP_DestroySession failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_RTSP_Destroy(&rtsp_context_ptr->pstRtspContext);
  if (ret != 0) {
    LOGE("CVI_RTSP_Destroy failed with %#x!\n", ret);
    return ret;
  }
  return ret;
}

int32_t TDL_WrapImage(TDLImage image, VIDEO_FRAME_INFO_S *frame) {
  TDLImageContext *image_context = (TDLImageContext *)image;
  *frame = *(VIDEO_FRAME_INFO_S *)image_context->image->getInternalData();
  return 0;
}

#ifdef __cplusplus
}
#endif