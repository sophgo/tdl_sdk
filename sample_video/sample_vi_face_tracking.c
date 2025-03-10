/**
 * This is a sample code for face tracking.
 */
#define LOG_TAG "SampleFaceTracking"
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

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *pstMWContext;
  cvitdl_service_handle_t stServiceHandle;
} SAMPLE_TDL_VENC_THREAD_ARG_S;

typedef struct {
  cvitdl_handle_t stTDLHandle;
  bool bTrackingWithFeature;
} SAMPLE_TDL_TDL_THREAD_ARG_S;

static cvtdl_face_t g_stFaceMeta = {0};
static cvtdl_tracker_t g_stTrackerMeta = {0};

void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.max_unmatched_num = 10;
  ds_conf->ktracker_conf.accreditation_threshold = 10;
  ds_conf->ktracker_conf.P_beta[2] = 0.1;
  ds_conf->ktracker_conf.P_beta[6] = 2.5e-2;
  ds_conf->kfilter_conf.Q_beta[2] = 0.1;
  ds_conf->kfilter_conf.Q_beta[6] = 2.5e-2;
}

cvtdl_service_brush_t get_random_brush(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand((uint32_t)seed);
  cvtdl_service_brush_t brush = {
      .color.r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .color.g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .color.b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .size = 2,
  };

  return brush;
}

void *run_venc(void *args) {
  printf("Enter encoder thread\n");
  SAMPLE_TDL_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_TDL_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvtdl_face_t stFaceMeta = {0};
  cvtdl_tracker_t stTrackerMeta = {0};

  cvtdl_service_brush_t stGreyBrush = CVI_TDL_Service_GetDefaultBrush();
  stGreyBrush.color.r = 105;
  stGreyBrush.color.g = 105;
  stGreyBrush.color.b = 105;

  cvtdl_service_brush_t stGreenBrush = CVI_TDL_Service_GetDefaultBrush();
  stGreenBrush.color.r = 0;
  stGreenBrush.color.g = 255;
  stGreenBrush.color.b = 0;

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      MutexAutoLock(ResultMutex, lock);
      CVI_TDL_CopyFaceMeta(&g_stFaceMeta, &stFaceMeta);
      CVI_TDL_CopyTrackerMeta(&g_stTrackerMeta, &stTrackerMeta);
    }

    // Draw different color for bbox accourding to tracker state.
    cvtdl_service_brush_t *brushes = malloc(stFaceMeta.size * sizeof(cvtdl_service_brush_t));
    for (uint32_t fid = 0; fid < stFaceMeta.size; fid++) {
      if (stTrackerMeta.info[fid].state == CVI_TRACKER_NEW) {
        brushes[fid] = stGreenBrush;
      } else if (stTrackerMeta.info[fid].state == CVI_TRACKER_UNSTABLE) {
        brushes[fid] = stGreyBrush;
      } else {  // CVI_TRACKER_STABLE
        brushes[fid] = get_random_brush(stFaceMeta.info[fid].unique_id, 64);
      }
    }

    // Fill name with unique id.
    for (uint32_t fid = 0; fid < stFaceMeta.size; fid++) {
      snprintf(stFaceMeta.info[fid].name, sizeof(stFaceMeta.info[fid].name), "UID: %" PRIu64 "",
               stFaceMeta.info[fid].unique_id);
    }

    s32Ret = CVI_TDL_Service_FaceDrawRect2(pstArgs->stServiceHandle, &stFaceMeta, &stFrame, true,
                                           brushes);
    if (s32Ret != CVI_TDL_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      printf("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_TDL_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
  error:
    free(brushes);
    CVI_TDL_Free(&stFaceMeta);
    CVI_TDL_Free(&stTrackerMeta);
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  printf("Exit encoder thread\n");
  pthread_exit(NULL);
}

void *run_tdl_thread(void *args) {
  printf("Enter TDL thread\n");
  SAMPLE_TDL_TDL_THREAD_ARG_S *pstTDLArgs = (SAMPLE_TDL_TDL_THREAD_ARG_S *)args;

  VIDEO_FRAME_INFO_S stFrame;
  cvtdl_face_t stFaceMeta = {0};
  cvtdl_tracker_t stTrackerMeta = {0};

  CVI_S32 s32Ret;
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);

    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    //*******************************************
    // Step 1: Face detection.
    GOTO_IF_FAILED(CVI_TDL_FaceDetection(pstTDLArgs->stTDLHandle, &stFrame,
                                         CVI_TDL_SUPPORTED_MODEL_RETINAFACE, &stFaceMeta),
                   s32Ret, inf_error);

    if (pstTDLArgs->bTrackingWithFeature) {
      // Step 2: Extract feature for all face in stFaceMeta.
      GOTO_IF_FAILED(CVI_TDL_FaceRecognition(pstTDLArgs->stTDLHandle, &stFrame, &stFaceMeta),
                     s32Ret, inf_error);
    }

    // Step 3: Multi-Object Tracking inference.
    GOTO_IF_FAILED(CVI_TDL_DeepSORT_Face(pstTDLArgs->stTDLHandle, &stFaceMeta, &stTrackerMeta),
                   s32Ret, inf_error);

    printf("face detected: %d\n", stFaceMeta.size);
    //*******************************************

    {
      MutexAutoLock(ResultMutex, lock);
      CVI_TDL_CopyFaceMeta(&stFaceMeta, &g_stFaceMeta);
      CVI_TDL_CopyTrackerMeta(&stTrackerMeta, &g_stTrackerMeta);
    }

  inf_error:
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    CVI_TDL_Free(&stFaceMeta);
    CVI_TDL_Free(&stTrackerMeta);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
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
  if (argc != 3 && argc != 2) {
    printf(
        "\nUsage: %s DET_MODEL_PATH [FR_MODEL_PATH]\n\n"
        "\tDET_MODEL_PATH, path to retinaface model\n"
        "\tFR_MODEL_PATH (optinal), path to face recognition model.\n",
        argv[0]);
    return -1;
  }

  if (CVI_MSG_Init()) {
		SAMPLE_PRT("CVI_MSG_Init fail\n");
		return 0;
	}

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

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
      .u32Width = 1920,
      .u32Height = 1080,
  };

  stMWConfig.stVBPoolConfig.u32VBPoolCount = 3;

  // VBPool 0 for VPSS Grp0 Chn0
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 3;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Height = stSensorSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Width = stSensorSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 1 for VPSS Grp0 Chn1
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 3;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 2 for TDL preprocessing
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_BGR_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = 768;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = 1024;
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

  GOTO_IF_FAILED(CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]),
                 s32Ret, setup_tdl_fail);

  bool bTrackingWithFeature = false;
  if (argc == 3) {
    // Tracking with face recognition features
    GOTO_IF_FAILED(CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION, argv[2]),
                   s32Ret, setup_tdl_fail);
    bTrackingWithFeature = true;
  }

  // Init DeepSORT
  CVI_TDL_DeepSORT_Init(stTDLHandle, true);
  cvtdl_deepsort_config_t ds_conf;
  CVI_TDL_DeepSORT_GetDefaultConfig(&ds_conf);
  set_sample_mot_config(&ds_conf);
  CVI_TDL_DeepSORT_SetConfig(stTDLHandle, &ds_conf, -1, false);

  pthread_t stVencThread, stTDLThread;
  SAMPLE_TDL_VENC_THREAD_ARG_S venc_args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  SAMPLE_TDL_TDL_THREAD_ARG_S ai_args = {.stTDLHandle = stTDLHandle,
                                         .bTrackingWithFeature = bTrackingWithFeature};

  pthread_create(&stVencThread, NULL, run_venc, &venc_args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, &ai_args);

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
