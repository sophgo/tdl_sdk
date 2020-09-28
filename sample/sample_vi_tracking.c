#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"

#include <cvi_ae.h>
#include <cvi_ae_comm.h>
#include <cvi_awb_comm.h>
#include <cvi_buffer.h>
#include <cvi_comm_isp.h>
#include <cvi_isp.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static volatile bool bExit = false;

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_t);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
  float mean[3];
  float factor[3];
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;

  if (strstr(model_name, "mobiledetv2")) {
    if (strcmp(model_name, "mobiledetv2-d0") == 0) {
      config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
      config->inference = CVI_AI_MobileDetV2_D0;
      config->input_size = 512;
    } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
      config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
      config->inference = CVI_AI_MobileDetV2_D1;
      config->input_size = 640;
    } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
      config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
      config->inference = CVI_AI_MobileDetV2_D2;
      config->input_size = 768;
    } else {
      ret = CVI_FAILURE;
    }
  } else if (strstr(model_name, "yolov3")) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;

    if (strcmp(model_name, "yolov3-320") == 0) {
      config->input_size = 320;
    } else if (strcmp(model_name, "yolov3-416") == 0) {
      config->input_size = 416;
    } else if (strcmp(model_name, "yolov3-608") == 0) {
      config->input_size = 608;
    } else {
      ret = CVI_FAILURE;
    }
  } else {
    ret = CVI_FAILURE;
  }

  return ret;
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

static CVI_S32 InitVI(const VI_PIPE viPipe, SAMPLE_VI_CONFIG_S *pstViConfig);

static CVI_S32 InitVO(const CVI_U32 width, const CVI_U32 height, SAMPLE_VO_CONFIG_S *stVoConfig);

static CVI_S32 InitVPSS(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                        const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                        const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened,
                        ModelConfig *model_config);

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf(
        "Usage: %s <detection_model_name> <detection_model_path> <reid_model_path> <open vo 1 or "
        "0>.\n"
        "\t model_name: detection model name should be one of {mobiledetv2-d0, mobiledetv2-d2, "
        "yolov3-416, yolov3-320}\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_BOOL isVoOpened = (strcmp(argv[4], "1") == 0) ? true : false;

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  ModelConfig model_config;
  if (createModelConfig(argv[1], &model_config) == CVI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVI_FAILURE;
  }

  CVI_S32 s32Ret = CVI_SUCCESS;
  //****************************************************************
  // Init VI, VO, Vpss
  VI_PIPE ViPipe = 0;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_CHN VpssChnVO = VPSS_CHN2;
  CVI_S32 GrpWidth = 1920;
  CVI_S32 GrpHeight = 1080;
  CVI_U32 VoLayer = 0;
  CVI_U32 VoChn = 0;
  SAMPLE_VI_CONFIG_S stViConfig;
  SAMPLE_VO_CONFIG_S stVoConfig;
  s32Ret = InitVI(ViPipe, &stViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video input failed with %d\n", s32Ret);
    return s32Ret;
  }

  const CVI_U32 voWidth = 1280;
  const CVI_U32 voHeight = 720;
  if (isVoOpened) {
    s32Ret = InitVO(voWidth, voHeight, &stVoConfig);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
      return s32Ret;
    }
    CVI_VO_HideChn(VoLayer, VoChn);
  }

  s32Ret = InitVPSS(VpssGrp, VpssChn, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe,
                    isVoOpened, &model_config);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }
  // Init end
  //****************************************************************

  cviai_handle_t facelib_handle = NULL;
  cviai_objservice_handle_t obj_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&facelib_handle, 1);
  ret |= CVI_AI_OBJService_CreateHandle(&obj_handle, facelib_handle);
  ret = CVI_AI_SetModelPath(facelib_handle, model_config.model_id, argv[2]);
  ret |= CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, model_config.model_id, false);
  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);

  // Create intersect area
  printf("Creating line intersect.\n");
  cvai_pts_t pts;
  pts.size = 2;
  pts.x = (float *)malloc(pts.size * sizeof(float));
  pts.y = (float *)malloc(pts.size * sizeof(float));
  pts.x[0] = 640;
  pts.y[0] = 0;
  pts.x[1] = 640;
  pts.y[1] = 719;
  CVI_AI_OBJService_SetIntersect(obj_handle, &pts);

  VIDEO_FRAME_INFO_S stFrame, stVOFrame;
  cvai_object_t obj_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stFrame, 1000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    //*******************************************
    // Tracking function calls.
    cvai_area_detect_e *status = NULL;
    // Step 1. Object detect inference.
    model_config.inference(facelib_handle, &stFrame, &obj_meta, CVI_DET_TYPE_PEOPLE);
    // Step 2. Object feature generator.
    CVI_AI_OSNet(facelib_handle, &stFrame, &obj_meta);
    // Step 3. Tracker.
    CVI_AI_Deepsort(facelib_handle, &obj_meta, &tracker_meta);
    // Step 4. Detect intersection.
    CVI_AI_OBJService_DetectIntersect(obj_handle, &stFrame, &obj_meta, &status);
    // Step 5. printf results.
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      printf("[%u][%" PRIu64 "] %s object state = %u, intersection = %u.\n", i,
             obj_meta.info[i].unique_id, obj_meta.info[i].name, tracker_meta.info[i].state,
             status[i]);
    }
    // Tracking function calls ends here.
    //*******************************************

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (isVoOpened) {
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_OBJService_DrawRect(&obj_meta, &stVOFrame);
      s32Ret = CVI_VO_SendFrame(VoLayer, VoChn, &stVOFrame, -1);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VO_SendFrame failed with %#x\n", s32Ret);
      }
      CVI_VO_ShowChn(VoLayer, VoChn);
      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&obj_meta);
    CVI_AI_Free(&tracker_meta);
    free(status);
  }

  CVI_AI_OBJService_DestroyHandle(obj_handle);
  CVI_AI_DestroyHandle(facelib_handle);

  // Exit vpss stuffs
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  abChnEnable[VpssChn] = CVI_TRUE;
  abChnEnable[VpssChnVO] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}

CVI_S32 InitVI(const VI_PIPE viPipe, SAMPLE_VI_CONFIG_S *pstViConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  SAMPLE_SNS_TYPE_E enSnsType = SONY_IMX307_MIPI_2M_30FPS_12BIT;
  WDR_MODE_E enWDRMode = WDR_MODE_NONE;
  DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
  PIXEL_FORMAT_E enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
  VI_VPSS_MODE_E enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;

  VI_CHN viChn = 0;
  VI_DEV viDev = 0;
  CVI_S32 s32WorkSnsId = 0;
  PIC_SIZE_E enPicSize;
  SIZE_S stSize;

  SAMPLE_COMM_VI_GetSensorInfo(pstViConfig);

  pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType = enSnsType;
  pstViConfig->s32WorkingViNum = 1;
  pstViConfig->as32WorkingViId[0] = 0;
  pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.MipiDev = 0xFF;
  pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.s32BusId = 3;
  pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.ViDev = viDev;
  pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.enWDRMode = enWDRMode;
  pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
  pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0] = viPipe;
  pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1] = -1;
  pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2] = -1;
  pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3] = -1;
  pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.ViChn = viChn;
  pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enPixFormat = enPixFormat;
  pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange = enDynamicRange;
  pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat = enVideoFormat;
  pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode = enCompressMode;

  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(enSnsType, &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = MMF_INIT_HELPER2(stSize.u32Width, stSize.u32Height, enPixFormat, 18, stSize.u32Width,
                            stSize.u32Height, enPixFormat, 18);
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

CVI_S32 InitVO(const CVI_U32 width, const CVI_U32 height, SAMPLE_VO_CONFIG_S *stVoConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  s32Ret = SAMPLE_COMM_VO_GetDefConfig(stVoConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
    return s32Ret;
  }
  RECT_S dispRect = {0, 0, height, width};
  SIZE_S imgSize = {height, width};

  stVoConfig->VoDev = 0;
  stVoConfig->enVoIntfType = VO_INTF_MIPI;
  stVoConfig->enIntfSync = VO_OUTPUT_720x1280_60;
  stVoConfig->stDispRect = dispRect;
  stVoConfig->stImageSize = imgSize;
  stVoConfig->enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
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
                 const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened,
                 ModelConfig *model_config) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

  abChnEnable[vpssChn] = CVI_TRUE;
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChn], voWidth, voHeight, PIXEL_FORMAT_RGB_888_PLANAR,
                          true);

  if (isVOOpened) {
    abChnEnable[vpssChnVO] = CVI_TRUE;
    VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChnVO], voWidth, voHeight,
                            PIXEL_FORMAT_YUV_PLANAR_420, true);
  }

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, grpWidth, grpHeight, PIXEL_FORMAT_YUV_PLANAR_420);

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
