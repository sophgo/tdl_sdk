#define LOG_TAG "SampleFD"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "middleware_utils.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <core/utils/vpss_helper.h>
#include <cvi_comm.h>
#include <rtsp.h>
#include <sample_comm.h>
#include "cvi_tdl.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile bool bExit = false;

static cvtdl_handpose21_meta_ts g_stHandKptMeta = {0};
static cvtdl_object_t g_stObjMeta = {0};

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *pstMWContext;
  cvitdl_service_handle_t stServiceHandle;
} SAMPLE_TDL_VENC_THREAD_ARG_S;

static const char *cls_name[] = {"fist", "five",  "four",   "none", "ok",
                                 "one",  "three", "three2", "two"};

void *run_venc(void *args) {
  printf("Enter encoder thread\n");
  SAMPLE_TDL_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_TDL_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvtdl_handpose21_meta_ts stHandKptMeta = {0};
  cvtdl_object_t stHandMeta = {0};

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    MutexAutoLock(ResultMutex, lock);
    memset(&stHandKptMeta, 0, sizeof(cvtdl_handpose21_meta_ts));
    memset(&stHandMeta, 0, sizeof(cvtdl_object_t));
    CVI_TDL_CopyHandPoses(&g_stHandKptMeta, &stHandKptMeta);
    CVI_TDL_CopyObjectMeta(&g_stObjMeta, &stHandMeta);
    if (stHandKptMeta.size > 0 && stHandMeta.size > 0) {
      CVI_TDL_Service_DrawHandKeypoint(pstArgs->stServiceHandle, &stFrame, &stHandKptMeta);
      cvtdl_service_brush_t brush_0 = {.size = 4, .color.r = 0, .color.g = 64, .color.b = 255};
      CVI_TDL_Service_ObjectDrawRect(pstArgs->stServiceHandle, &stHandMeta, &stFrame, false,
                                     brush_0);
      char *id_num = calloc(64, sizeof(char));
      for (uint32_t i = 0; i < stHandKptMeta.size; i++) {
        sprintf(id_num, "cls:%s, score:%f", cls_name[stHandKptMeta.info[i].label],
                stHandKptMeta.info[i].score);
        CVI_TDL_Service_ObjectWriteText(id_num, stHandKptMeta.info[i].bbox_x,
                                        stHandKptMeta.info[i].bbox_y, &stFrame, 1, 1, 1);
      }
      free(id_num);
    }
    CVI_TDL_Free(&stHandKptMeta);

    s32Ret = SAMPLE_TDL_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
    if (s32Ret != CVI_SUCCESS) {
      goto error;
    }
  error:
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  printf("Exit encoder thread\n");
  pthread_exit(NULL);
}

void *run_tdl_thread(void *pHandle) {
  printf("Enter TDL thread\n");
  cvitdl_handle_t pstTDLHandle = (cvitdl_handle_t)pHandle;

  VIDEO_FRAME_INFO_S stFrame;
  cvtdl_object_t stHandMeta = {0};
  cvtdl_handpose21_meta_ts stHandKptMeta = {0};

  float buffer[42];
  CVI_S32 s32Ret;
  static uint32_t count = 0;

  while (bExit == false) {
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    count++;

    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    memset(&stHandMeta, 0, sizeof(cvtdl_object_t));
    memset(&stHandKptMeta, 0, sizeof(cvtdl_handpose21_meta_ts));

    s32Ret = CVI_TDL_Detection(pstTDLHandle, &stFrame, CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION,
                               &stHandMeta);
    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }
    CVI_TDL_CopyObjectMeta(&stHandMeta, &g_stObjMeta);

    stHandKptMeta.size = stHandMeta.size;
    stHandKptMeta.width = stHandMeta.width;
    stHandKptMeta.height = stHandMeta.height;
    stHandKptMeta.info =
        (cvtdl_handpose21_meta_t *)malloc(sizeof(cvtdl_handpose21_meta_t) * (stHandMeta.size));

    for (int i = 0; i < stHandMeta.size; i++) {
      stHandKptMeta.info[i].bbox_x = stHandMeta.info[i].bbox.x1;
      stHandKptMeta.info[i].bbox_y = stHandMeta.info[i].bbox.y1;
      stHandKptMeta.info[i].bbox_w = stHandMeta.info[i].bbox.x2 - stHandMeta.info[i].bbox.x1;
      stHandKptMeta.info[i].bbox_h = stHandMeta.info[i].bbox.y2 - stHandMeta.info[i].bbox.y1;
    }
    s32Ret = CVI_TDL_HandKeypoint(pstTDLHandle, &stFrame, &stHandKptMeta);
    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("keypoint inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    for (uint32_t i = 0; i < stHandKptMeta.size; i++) {
      for (uint32_t j = 0; j < 42; j++) {
        if (j % 2 == 0) {
          buffer[j] = stHandKptMeta.info[i].xn[j / 2];
        } else {
          buffer[j] = stHandKptMeta.info[i].yn[j / 2];
        }
      }
      VIDEO_FRAME_INFO_S Frame;
      Frame.stVFrame.pu8VirAddr[0] = buffer;  // Global buffer
      Frame.stVFrame.u32Height = 1;
      Frame.stVFrame.u32Width = 42 * sizeof(float);
      CVI_TDL_HandKeypointClassification(pstTDLHandle, &Frame, &stHandKptMeta.info[i]);
    }
    MutexAutoLock(ResultMutex, lock);
    CVI_TDL_CopyHandPoses(&stHandKptMeta, &g_stHandKptMeta);

  inf_error:
    CVI_TDL_Free(&stHandMeta);
    CVI_TDL_Free(&stHandKptMeta);
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
    gettimeofday(&t1, NULL);
    uint64_t execution_time = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec);
    if (count % 100 == 0) {
      printf("ai thread execution time: %.2f(ms)\n", (float)execution_time / 1000.);
    }
  }

  printf("Exit TDL thread\n");
  pthread_exit(NULL);
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);
  printf("handle signal, signo: %d\n", signo);
  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  CVI_SYS_Exit();
  CVI_VB_Exit();
  if (argc != 4) {
    printf(
        "\nUsage: %s HAND_DETECTION_MODEL_PATH.\n\n"
        "\tHAND_KEY_POINT_MODEL_PATH, HAND_KEY_POINT_CLS_MODEL_PATH.\n",
        argv[0]);
    return CVI_TDL_FAILURE;
  }
  if (CVI_MSG_Init()) {
		SAMPLE_PRT("CVI_MSG_Init fail\n");
		return 0;
	}

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_BOOL abChnEnable[VPSS_MAX_CHN_NUM] = {
      CVI_TRUE,
  };
  for (VPSS_GRP VpssGrp = 0; VpssGrp < VPSS_MAX_GRP_NUM; ++VpssGrp)
    SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_TDL_MW_CONFIG_S stMWConfig = {0};

  CVI_S32 s32Ret = SAMPLE_TDL_Get_VI_Config(&stMWConfig.stViConfig);
  if (s32Ret != CVI_SUCCESS || stMWConfig.stViConfig.s32WorkingViNum <= 0) {
    printf("Failed to get senor infomation from ini file (/mnt/data/sensor_cfg.ini).\n");
    return -1;
  }

  // Get VI size
  PIC_SIZE_E enPicSize;
  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stMWConfig.stViConfig.astViInfo[0].stSnsInfo.enSnsType,
                                          &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("Cannot get senor size\n");
    return -1;
  }

  SIZE_S stSensorSize;
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSensorSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("Cannot get senor size\n");
    return -1;
  }

  // Setup frame size of video encoder to 1080p
  SIZE_S stVencSize = {
      .u32Width = 1280,
      .u32Height = 720,
  };

  stMWConfig.stVBPoolConfig.u32VBPoolCount = 3;

  // VBPool 0 for VPSS Grp0 Chn0
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 2;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Height = stSensorSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Width = stSensorSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 1 for VPSS Grp0 Chn1
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 2;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 2 for TDL preprocessing
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_BGR_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = stVencSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = stVencSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].bBind = false;

  // Setup VPSS Grp0
  stMWConfig.stVPSSPoolConfig.u32VpssGrpCount = 1;
#ifndef __CV186X__
  stMWConfig.stVPSSPoolConfig.stVpssMode.aenInput[0] = VPSS_INPUT_MEM;
  stMWConfig.stVPSSPoolConfig.stVpssMode.enMode = VPSS_MODE_DUAL;
  stMWConfig.stVPSSPoolConfig.stVpssMode.ViPipe[0] = 0;
  stMWConfig.stVPSSPoolConfig.stVpssMode.aenInput[1] = VPSS_INPUT_ISP;
  stMWConfig.stVPSSPoolConfig.stVpssMode.ViPipe[1] = 0;
#endif

  SAMPLE_TDL_VPSS_CONFIG_S *pstVpssConfig = &stMWConfig.stVPSSPoolConfig.astVpssConfig[0];
  pstVpssConfig->bBindVI = true;

  // Assign device 1 to VPSS Grp0, because device1 has 3 outputs in dual mode.
  VPSS_GRP_DEFAULT_HELPER2(&pstVpssConfig->stVpssGrpAttr, stSensorSize.u32Width,
                           stSensorSize.u32Height, VI_PIXEL_FORMAT, 1);
  pstVpssConfig->u32ChnCount = 2;
  pstVpssConfig->u32ChnBindVI = 0;
  VPSS_CHN_DEFAULT_HELPER(&pstVpssConfig->astVpssChnAttr[0], stVencSize.u32Width,
                          stVencSize.u32Height, VI_PIXEL_FORMAT, true);
  VPSS_CHN_DEFAULT_HELPER(&pstVpssConfig->astVpssChnAttr[1], stVencSize.u32Width,
                          stVencSize.u32Height, VI_PIXEL_FORMAT, true);

  // Get default VENC configurations
  SAMPLE_TDL_Get_Input_Config(&stMWConfig.stVencConfig.stChnInputCfg);
  stMWConfig.stVencConfig.u32FrameWidth = stVencSize.u32Width;
  stMWConfig.stVencConfig.u32FrameHeight = stVencSize.u32Height;

  // Get default RTSP configurations
  SAMPLE_TDL_Get_RTSP_Config(&stMWConfig.stRTSPConfig.stRTSPConfig);

  SAMPLE_TDL_MW_CONTEXT stMWContext = {0};
  s32Ret = SAMPLE_TDL_Init_WM(&stMWConfig, &stMWContext);
  if (s32Ret != CVI_SUCCESS) {
    printf("init middleware failed! ret=%x\n", s32Ret);
    return -1;
  }

  cvitdl_handle_t stTDLHandle = NULL;

  // Create TDL handle and assign VPSS Grp1 Device 0 to TDL SDK
  GOTO_IF_FAILED(CVI_TDL_CreateHandle2(&stTDLHandle, 1, 0), s32Ret, create_tdl_fail);

  GOTO_IF_FAILED(CVI_TDL_SetVBPool(stTDLHandle, 0, 2), s32Ret, create_service_fail);

  CVI_TDL_SetVpssTimeout(stTDLHandle, 1000);

  cvitdl_service_handle_t stServiceHandle = NULL;
  GOTO_IF_FAILED(CVI_TDL_Service_CreateHandle(&stServiceHandle, stTDLHandle), s32Ret,
                 create_service_fail);
  GOTO_IF_FAILED(CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION, argv[1]),
                 s32Ret, setup_tdl_fail);
  GOTO_IF_FAILED(CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_HAND_KEYPOINT, argv[2]),
                 s32Ret, setup_tdl_fail);
  GOTO_IF_FAILED(
      CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION, argv[3]),
      s32Ret, setup_tdl_fail);
  CVI_TDL_SetModelThreshold(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION, 0.55);
  pthread_t stVencThread, stTDLThread;
  SAMPLE_TDL_VENC_THREAD_ARG_S args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  pthread_create(&stVencThread, NULL, run_venc, &args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, stTDLHandle);

  pthread_join(stVencThread, NULL);
  pthread_join(stTDLThread, NULL);

setup_tdl_fail:
  CVI_TDL_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_TDL_DestroyHandle(stTDLHandle);
create_tdl_fail:
  SAMPLE_TDL_Destroy_MW(&stMWContext);

	CVI_MSG_Deinit();
  return 0;
}
