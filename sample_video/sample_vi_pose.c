#include "middleware_utils.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"
#include <rtsp.h>
#include <sample_comm.h>
#include "cvi_tdl.h"

#include <core/utils/vpss_helper.h>
#include <cvi_comm.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <sys/time.h>


static volatile bool bExit = false;
static volatile bool init_alg_param = false;

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *pstMWContext;
  cvitdl_service_handle_t stServiceHandle;
} SAMPLE_TDL_VENC_THREAD_ARG_S;

typedef struct {
  ODInferenceFunc object_detect;
  CVI_TDL_SUPPORTED_MODEL_E enOdModelId;
  cvitdl_handle_t stTDLHandle;
  bool bTrackingWithFeature;
  int smooth_type;
} SAMPLE_TDL_TDL_THREAD_ARG_S;

static cvtdl_object_t g_stObjMeta = {0};
static cvtdl_tracker_t g_stTrackerMeta = {0};

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.P_beta[2] = 0.01;
  ds_conf->ktracker_conf.P_beta[6] = 1e-5;

  // ds_conf.kfilter_conf.Q_beta[2] = 0.1;
  ds_conf->kfilter_conf.Q_beta[2] = 0.01;
  ds_conf->kfilter_conf.Q_beta[6] = 1e-5;
  ds_conf->kfilter_conf.R_beta[2] = 0.1;
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
  SAMPLE_TDL_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_TDL_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvtdl_object_t stObjMeta = {0};
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
      CVI_TDL_CopyObjectMeta(&g_stObjMeta, &stObjMeta);
      CVI_TDL_CopyTrackerMeta(&g_stTrackerMeta, &stTrackerMeta);
    }

    // Draw different color for bbox accourding to tracker state.
    cvtdl_service_brush_t *brushes = malloc(stObjMeta.size * sizeof(cvtdl_service_brush_t));
    for (uint32_t oid = 0; oid < stObjMeta.size; oid++) {
      if (stTrackerMeta.info[oid].state == CVI_TRACKER_NEW) {
        brushes[oid] = stGreenBrush;
      } else if (stTrackerMeta.info[oid].state == CVI_TRACKER_UNSTABLE) {
        brushes[oid] = stGreyBrush;
      } else {  // CVI_TRACKER_STABLE
        brushes[oid] = get_random_brush(stObjMeta.info[oid].unique_id, 64);
      }
    }

    for (uint32_t oid = 0; oid < stObjMeta.size; oid++) {
      snprintf(stObjMeta.info[oid].name, sizeof(stObjMeta.info[oid].name),
               "UID: %" PRIu64 "",
               stObjMeta.info[oid].unique_id);
    }

    s32Ret = CVI_TDL_Service_ObjectDrawRect2(pstArgs->stServiceHandle, &stObjMeta, &stFrame, true,
                                            brushes);

    CVI_TDL_Service_ObjectDrawPose(&stObjMeta, &stFrame, 0.3);

    if (s32Ret != CVI_TDL_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      printf("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_TDL_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
  error:
    free(brushes);
    CVI_TDL_Free(&stObjMeta);
    CVI_TDL_Free(&stTrackerMeta);
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  pthread_exit(NULL);
}

void *run_ai_thread(void *args) {
  SAMPLE_TDL_TDL_THREAD_ARG_S *pstTDLArgs = (SAMPLE_TDL_TDL_THREAD_ARG_S *)args;

  VIDEO_FRAME_INFO_S stFrame;
  cvtdl_object_t stObjMeta = {0};
  cvtdl_tracker_t stTrackerMeta = {0};

  CVI_S32 s32Ret;

  size_t counter = 0;
  uint32_t last_time_ms = get_time_in_ms();
  size_t last_counter = 0;

  while (bExit == false) {
    counter += 1;

    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);

    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    int frm_diff = counter - last_counter;
    if (frm_diff > 20) {
      uint32_t cur_ts_ms = get_time_in_ms();
      float fps = frm_diff * 1000.0 / (cur_ts_ms - last_time_ms);
      // CVI_TDL_Set_Fall_FPS(pstTDLArgs->stTDLHandle, fps);  // set only once if fps is stable
      last_time_ms = cur_ts_ms;
      last_counter = counter;
      printf("++++++++++++ frame:%d,fps:%.2f\n", (int)counter, fps);
    }

    if(!init_alg_param){
      SmoothAlgParam smooth_alg_param = CVI_TDL_Get_Smooth_Algparam(pstTDLArgs->stTDLHandle);
      smooth_alg_param.image_width = stFrame.stVFrame.u32Width;
      smooth_alg_param.image_height = stFrame.stVFrame.u32Height;
      smooth_alg_param.smooth_type = pstTDLArgs->smooth_type;
      CVI_TDL_Set_Smooth_Algparam(pstTDLArgs->stTDLHandle, smooth_alg_param);
      init_alg_param = true;
    }

    //*******************************************
    // Step 1: Object detect inference.
    s32Ret = pstTDLArgs->object_detect(pstTDLArgs->stTDLHandle, &stFrame, pstTDLArgs->enOdModelId, &stObjMeta);
    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    // Step 2: Multi-Object Tracking inference.
    s32Ret = CVI_TDL_DeepSORT_Obj(pstTDLArgs->stTDLHandle, &stObjMeta, &stTrackerMeta,
                                 pstTDLArgs->bTrackingWithFeature);
    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    // Step 3: Smooth keypoints inference.
    s32Ret = CVI_TDL_Smooth_Keypoints(pstTDLArgs->stTDLHandle, &stObjMeta);
    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }


    {
      MutexAutoLock(ResultMutex, lock);
      CVI_TDL_CopyObjectMeta(&stObjMeta, &g_stObjMeta);
      CVI_TDL_CopyTrackerMeta(&stTrackerMeta, &g_stTrackerMeta);
    }

    CVI_TDL_Free(&stObjMeta);
    CVI_TDL_Free(&stTrackerMeta);

  inf_error:
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    CVI_TDL_Free(&stObjMeta);
    CVI_TDL_Free(&stTrackerMeta);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }

  pthread_exit(NULL);
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);
  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf(
        "\nUsage: %s DET_MODEL_PATH SMOOTH_TYPE \n\n"
        "\tDET_MODEL_PATH, path to keypoints detection model\n \tSMOOTH_TYPE, should be 0 or 1 (defult to 0)\n",
        argv[0]);
    return -1;
  }

  CVI_TDL_SUPPORTED_MODEL_E enOdModelId = CVI_TDL_SUPPORTED_MODEL_YOLOV8POSE;

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

  GOTO_IF_FAILED(CVI_TDL_OpenModel(stTDLHandle, enOdModelId, argv[1]), s32Ret, setup_tdl_fail);

  bool bTrackingWithFeature = false;

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

  SAMPLE_TDL_TDL_THREAD_ARG_S ai_args = {
      .enOdModelId = enOdModelId,
      .object_detect = CVI_TDL_PoseDetection,
      .stTDLHandle = stTDLHandle,
      .bTrackingWithFeature = bTrackingWithFeature,
      .smooth_type = atoi(argv[2])
  };

  pthread_create(&stVencThread, NULL, run_venc, &venc_args);
  pthread_create(&stTDLThread, NULL, run_ai_thread, &ai_args);

  pthread_join(stVencThread, NULL);
  pthread_join(stTDLThread, NULL);

setup_tdl_fail:
  CVI_TDL_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_TDL_DestroyHandle(stTDLHandle);
create_tdl_fail:
  SAMPLE_TDL_Destroy_MW(&stMWContext);

  return 0;
}