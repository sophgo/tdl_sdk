#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cvi_ae.h"
#include "cvi_ae_comm.h"
#include "cvi_awb_comm.h"
#include "cvi_buffer.h"
#include "cvi_comm_isp.h"
#include "cvi_isp.h"
#include "cvi_sys.h"
#include "cvi_vb.h"
#include "cvi_vi.h"

#include "cviai.h"
#include "sample_comm.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static volatile bool bExit = false;

cviai_handle_t facelib_handle = NULL;
static SAMPLE_VI_CONFIG_S stViConfig;
// static	SAMPLE_VO_CONFIG_S stVoConfig;

static VI_PIPE ViPipe = 0;
static VPSS_GRP VpssGrp = 0;
static VPSS_CHN VpssChnFR = VPSS_CHN0;
static VPSS_CHN VpssChnVO = VPSS_CHN1;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;
// static CVI_U32 VoLayer = 0;
// static CVI_U32 VoChn = 0;

static int GetVideoframe(VIDEO_FRAME_INFO_S *stfrFrame, VIDEO_FRAME_INFO_S *stVOFrame) {
  int s32Ret = CVI_SUCCESS;

  s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnFR, stfrFrame, 1000);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, stVOFrame, 1000);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}

static int ReleaseVideoframe(VIDEO_FRAME_INFO_S *stfrFrame, VIDEO_FRAME_INFO_S *stVOFrame) {
  int s32Ret = CVI_SUCCESS;

  s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnFR, stfrFrame);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
    return s32Ret;
  }

  s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, stVOFrame);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
    return s32Ret;
  }

  return s32Ret;
}

static int DoFd(cviai_handle_t facelib_handle, VIDEO_FRAME_INFO_S *stfrFrame, cvai_face_t *face) {
  int face_count = 0;

  CVI_AI_RetinaFace(facelib_handle, stfrFrame, face, &face_count);
  printf("face_count %d\n", face->size);
  CVI_AI_FaceAttribute(facelib_handle, stfrFrame, face);

  return 0;
}

static void Exit() {
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChnFR, VpssGrp);

  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  abChnEnable[VpssChnFR] = CVI_TRUE;
  abChnEnable[VpssChnVO] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}

static void Run() {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VIDEO_FRAME_INFO_S stfrFrame, stVOFrame;
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));

  while (bExit == false) {
    s32Ret = GetVideoframe(&stfrFrame, &stVOFrame);
    if (s32Ret != CVI_SUCCESS) {
      break;
    }

    DoFd(facelib_handle, &stfrFrame, &face);

    CVI_AI_FRService_DrawRect(&face, &stVOFrame);

    // s32Ret = CVI_VO_SendFrame(VoLayer, VoChn, &stVOFrame, -1);
    // if (s32Ret != CVI_SUCCESS) {
    // 	printf("CVI_VO_SendFrame failed with %#x\n", s32Ret);
    // }
    // CVI_VO_ShowChn(VoLayer,VoChn);

    s32Ret = ReleaseVideoframe(&stfrFrame, &stVOFrame);
    if (s32Ret != CVI_SUCCESS) {
      break;
    }

    CVI_AI_Free(&face);
  }
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

static void SetVIConfig(SAMPLE_VI_CONFIG_S *stViConfig) {
  CVI_S32 s32WorkSnsId = 0;

  SAMPLE_SNS_TYPE_E enSnsType = SONY_IMX307_MIPI_2M_30FPS_12BIT;
  WDR_MODE_E enWDRMode = WDR_MODE_NONE;
  DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
  PIXEL_FORMAT_E enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
  VI_VPSS_MODE_E enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;

  SAMPLE_COMM_VI_GetSensorInfo(stViConfig);

  stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType = enSnsType;
  stViConfig->s32WorkingViNum = 1;
  stViConfig->as32WorkingViId[0] = 0;
  stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.MipiDev = 0xFF;
  stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.s32BusId = 3;
  stViConfig->astViInfo[s32WorkSnsId].stDevInfo.ViDev = 0;
  stViConfig->astViInfo[s32WorkSnsId].stDevInfo.enWDRMode = enWDRMode;
  stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
  stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0] = ViPipe;
  stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1] = -1;
  stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2] = -1;
  stViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3] = -1;
  stViConfig->astViInfo[s32WorkSnsId].stChnInfo.ViChn = 0;
  stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enPixFormat = enPixFormat;
  stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange = enDynamicRange;
  stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat = enVideoFormat;
  stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode = enCompressMode;
}

// static CVI_S32 SetVoConfig(SAMPLE_VO_CONFIG_S* stVoConfig)
// {
// 	RECT_S stDefDispRect  = {0, 0, 720, 1280};
// 	SIZE_S stDefImageSize = {720, 1280};
// 	CVI_S32 s32Ret = CVI_SUCCESS;
// 	s32Ret = SAMPLE_COMM_VO_GetDefConfig(stVoConfig);
// 	if (s32Ret != CVI_SUCCESS) {
// 		printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
// 		return s32Ret;
// 	}

// 	stVoConfig->VoDev	 = 0;
// 	stVoConfig->enVoIntfType  = VO_INTF_MIPI;
// 	stVoConfig->enIntfSync    = VO_OUTPUT_720x1280_60;
// 	stVoConfig->stDispRect	 = stDefDispRect;
// 	stVoConfig->stImageSize	 = stDefImageSize;
// 	stVoConfig->enPixFormat	 = PIXEL_FORMAT_YUV_PLANAR_420;
// 	stVoConfig->enVoMode	 = VO_MODE_1MUX;
// 	return s32Ret;
// }

static CVI_S32 InitVPSS() {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

  abChnEnable[VpssChnFR] = CVI_TRUE;
  stVpssChnAttr[VpssChnFR].u32Width = 640;
  stVpssChnAttr[VpssChnFR].u32Height = 480;
  stVpssChnAttr[VpssChnFR].enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr[VpssChnFR].enPixelFormat = PIXEL_FORMAT_BGR_888;
  stVpssChnAttr[VpssChnFR].stFrameRate.s32SrcFrameRate = 30;
  stVpssChnAttr[VpssChnFR].stFrameRate.s32DstFrameRate = 30;
  stVpssChnAttr[VpssChnFR].u32Depth = 1;
  stVpssChnAttr[VpssChnFR].bMirror = CVI_FALSE;
  stVpssChnAttr[VpssChnFR].bFlip = CVI_FALSE;
  stVpssChnAttr[VpssChnFR].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  stVpssChnAttr[VpssChnFR].stAspectRatio.bEnableBgColor = CVI_TRUE;
  stVpssChnAttr[VpssChnFR].stAspectRatio.u32BgColor = COLOR_RGB_BLACK;
  stVpssChnAttr[VpssChnFR].stNormalize.bEnable = CVI_FALSE;

  abChnEnable[VpssChnVO] = CVI_TRUE;
  stVpssChnAttr[VpssChnVO].u32Width = 1280;
  stVpssChnAttr[VpssChnVO].u32Height = 720;
  stVpssChnAttr[VpssChnVO].enVideoFormat = VIDEO_FORMAT_LINEAR;
  stVpssChnAttr[VpssChnVO].enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  stVpssChnAttr[VpssChnVO].stFrameRate.s32SrcFrameRate = 30;
  stVpssChnAttr[VpssChnVO].stFrameRate.s32DstFrameRate = 30;
  stVpssChnAttr[VpssChnVO].u32Depth = 1;
  stVpssChnAttr[VpssChnVO].bMirror = CVI_FALSE;
  stVpssChnAttr[VpssChnVO].bFlip = CVI_FALSE;
  // stVpssChnAttr[VpssChnVO].stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  stVpssChnAttr[VpssChnVO].stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
  stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.s32X = 0;
  stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.s32Y = 0;
  stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.u32Width = 1280;
  stVpssChnAttr[VpssChnVO].stAspectRatio.stVideoRect.u32Height = 720;
  stVpssChnAttr[VpssChnVO].stAspectRatio.bEnableBgColor = CVI_TRUE;
  stVpssChnAttr[VpssChnVO].stNormalize.bEnable = CVI_FALSE;

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
  stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
  stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  stVpssGrpAttr.u32MaxW = vpssgrp_width;
  stVpssGrpAttr.u32MaxH = vpssgrp_height;
  // only for test here. u8VpssDev should be decided by VPSS_MODE and usage.
  stVpssGrpAttr.u8VpssDev = 0;

  /*start vpss*/
  s32Ret = SAMPLE_COMM_VPSS_Init(VpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe, VpssChnFR, VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}

static CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *stViConfig) {
  VB_CONFIG_S stVbConf;
  PIC_SIZE_E enPicSize;
  CVI_U32 u32BlkSize, u32BlkRotSize, u32BlkRGBSize;
  SIZE_S stSize;
  CVI_S32 s32Ret = CVI_SUCCESS;

  VI_DEV ViDev = 0;
  CVI_S32 s32WorkSnsId = 0;
  VI_PIPE_ATTR_S stPipeAttr;

  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType,
                                          &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
    return s32Ret;
  }

  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = 2;

  u32BlkSize = COMMON_GetPicBufferSize(
      stSize.u32Width, stSize.u32Height, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8,
      stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  u32BlkRotSize = COMMON_GetPicBufferSize(
      stSize.u32Height, stSize.u32Width, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8,
      stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  u32BlkSize = MAX(u32BlkSize, u32BlkRotSize);
  u32BlkRGBSize = COMMON_GetPicBufferSize(
      vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_BGR_888, DATA_BITWIDTH_8,
      stViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode, DEFAULT_ALIGN);
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = 32;
  stVbConf.astCommPool[0].enRemapMode = VB_REMAP_MODE_NOCACHE;
  stVbConf.astCommPool[1].u32BlkSize = u32BlkRGBSize;
  stVbConf.astCommPool[1].u32BlkCnt = 2;
  stVbConf.astCommPool[1].enRemapMode = VB_REMAP_MODE_NOCACHE;
  printf("common pool[0] BlkSize %d\n", u32BlkSize);

  s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    printf("system init failed with %#x\n", s32Ret);
    return -1;
  }

  s32Ret = SAMPLE_COMM_VI_StartSensor(stViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("system start sensor failed with %#x\n", s32Ret);
    return s32Ret;
  }
  SAMPLE_COMM_VI_StartDev(&stViConfig->astViInfo[ViDev]);
  SAMPLE_COMM_VI_StartMIPI(stViConfig);

  memset(&stPipeAttr, 0, sizeof(VI_PIPE_ATTR_S));
  stPipeAttr.bYuvSkip = CVI_FALSE;
  stPipeAttr.u32MaxW = stSize.u32Width;
  stPipeAttr.u32MaxH = stSize.u32Height;
  stPipeAttr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP;
  stPipeAttr.enBitWidth = DATA_BITWIDTH_12;
  stPipeAttr.stFrameRate.s32SrcFrameRate = -1;
  stPipeAttr.stFrameRate.s32DstFrameRate = -1;
  stPipeAttr.bNrEn = CVI_TRUE;
  s32Ret = CVI_VI_CreatePipe(ViPipe, &stPipeAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VI_CreatePipe failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_VI_StartPipe(ViPipe);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VI_StartPipe failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_VI_GetPipeAttr(ViPipe, &stPipeAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VI_StartPipe failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_CreateIsp(stViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("VI_CreateIsp failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  return SAMPLE_COMM_VI_StartViChn(&stViConfig->astViInfo[ViDev]);
}

// int InitVO(SAMPLE_VO_CONFIG_S* stVoConfig)
// {
// 	CVI_S32 s32Ret = CVI_SUCCESS;

// 	s32Ret = SAMPLE_COMM_VO_StartVO(stVoConfig);
// 	if (s32Ret != CVI_SUCCESS) {
// 		printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", s32Ret);
// 	}

// 	printf("SAMPLE_COMM_VO_StartVO done\n");
// 	return s32Ret;
// }

int main(void) {
  CVI_S32 s32Ret = CVI_SUCCESS;

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  SetVIConfig(&stViConfig);
  s32Ret = InitVI(&stViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video input failed with %d\n", s32Ret);
    return s32Ret;
  }

  // s32Ret = SetVoConfig(&stVoConfig);
  // if (s32Ret != CVI_SUCCESS) {
  // 	printf("SetVoConfig failed with %d\n", s32Ret);
  // 	return s32Ret;
  // }

  // s32Ret = InitVO(&stVoConfig);
  // if (s32Ret != CVI_SUCCESS) {
  // 	printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
  // 	return s32Ret;
  // }
  // CVI_VO_HideChn(VoLayer,VoChn);

  s32Ret = InitVPSS();
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }

  int ret = CVI_AI_CreateHandle(&facelib_handle);
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                            "/mnt/data/retina_face.cvimodel");
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                            "/mnt/data/bmface.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }
  // Do vpss frame transform in retina face
  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  Run();

  CVI_AI_DestroyHandle(facelib_handle);
  Exit();
}
