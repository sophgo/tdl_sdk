/**
 * This is a sample code for object tracking.
 */
#define LOG_TAG "SampleObjectTracking"
#define LOG_LEVEL LOG_LEVEL_INFO

#include "middleware_utils.h"
#include "sample_log.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <core/utils/vpss_helper.h>
#include <cvi_comm.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <cviai.h>
#include <rtsp.h>
#include <sample_comm.h>

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static volatile bool bExit = false;

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_AI_MW_CONTEXT *pstMWContext;
  cviai_service_handle_t stServiceHandle;
} SAMPLE_AI_VENC_THREAD_ARG_S;

typedef struct {
  ODInferenceFunc object_detect;
  CVI_AI_SUPPORTED_MODEL_E enOdModelId;
  cviai_handle_t stAIHandle;
  bool bTrackingWithFeature;
  int img_num;
} SAMPLE_AI_AI_THREAD_ARG_S;

static cvai_object_t g_stObjMeta = {0};
static cvai_tracker_t g_stTrackerMeta = {0};

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void set_sample_mot_config(cvai_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.P_beta[2] = 0.01;
  ds_conf->ktracker_conf.P_beta[6] = 1e-5;

  // ds_conf.kfilter_conf.Q_beta[2] = 0.1;
  ds_conf->kfilter_conf.Q_beta[2] = 0.01;
  ds_conf->kfilter_conf.Q_beta[6] = 1e-5;
  ds_conf->kfilter_conf.R_beta[2] = 0.1;
}

cvai_service_brush_t get_random_brush(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand((uint32_t)seed);
  cvai_service_brush_t brush = {
      .color.r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .color.g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .color.b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min,
      .size = 2,
  };

  return brush;
}

void *run_venc(void *args) {
  AI_LOGI("Enter encoder thread\n");
  SAMPLE_AI_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_AI_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvai_object_t stObjMeta = {0};
  cvai_tracker_t stTrackerMeta = {0};

  cvai_service_brush_t stGreyBrush = CVI_AI_Service_GetDefaultBrush();
  stGreyBrush.color.r = 105;
  stGreyBrush.color.g = 105;
  stGreyBrush.color.b = 105;

  cvai_service_brush_t stGreenBrush = CVI_AI_Service_GetDefaultBrush();
  stGreenBrush.color.r = 0;
  stGreenBrush.color.g = 255;
  stGreenBrush.color.b = 0;

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      AI_LOGE("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      MutexAutoLock(ResultMutex, lock);
      CVI_AI_CopyObjectMeta(&g_stObjMeta, &stObjMeta);
      CVI_AI_CopyTrackerMeta(&g_stTrackerMeta, &stTrackerMeta);
    }

    // Draw different color for bbox accourding to tracker state.
    cvai_service_brush_t *brushes = malloc(stObjMeta.size * sizeof(cvai_service_brush_t));
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
               "fall: [%d], UID: %" PRIu64 "", (int)stObjMeta.info[oid].pedestrian_properity->fall,
               stObjMeta.info[oid].unique_id);
    }

    s32Ret = CVI_AI_Service_ObjectDrawRect2(pstArgs->stServiceHandle, &stObjMeta, &stFrame, true,
                                            brushes);

    // char text[256] = {0};
    // if (stObjMeta.size > 0) {
    //   sprintf(text, "id:%ld, a:%.1f,r:%.2f,s:%.1f,m:%d,st:%d", stObjMeta.info[0].unique_id,
    //           stObjMeta.info[0].human_angle, stObjMeta.info[0].aspect_ratio,
    //           stObjMeta.info[0].speed, stObjMeta.info[0].is_moving, stObjMeta.info[0].status);
    // }

    // CVI_AI_Service_ObjectWriteText(text, 20, 100, &stFrame, 255, 0, 0);

    if (s32Ret != CVIAI_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      AI_LOGE("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_AI_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
  error:
    free(brushes);
    CVI_AI_Free(&stObjMeta);
    CVI_AI_Free(&stTrackerMeta);
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  AI_LOGI("Exit encoder thread\n");
  pthread_exit(NULL);
}

void *run_ai_thread(void *args) {
  AI_LOGI("Enter AI thread\n");
  SAMPLE_AI_AI_THREAD_ARG_S *pstAIArgs = (SAMPLE_AI_AI_THREAD_ARG_S *)args;

  VIDEO_FRAME_INFO_S stFrame;
  cvai_object_t stObjMeta = {0};
  cvai_tracker_t stTrackerMeta = {0};

  CVI_S32 s32Ret;

  size_t counter = 0;
  uint32_t last_time_ms = get_time_in_ms();
  size_t last_counter = 0;

  while (bExit == false) {
    counter += 1;

    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);

    if (s32Ret != CVI_SUCCESS) {
      AI_LOGE("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    int frm_diff = counter - last_counter;
    if (frm_diff > 20) {
      uint32_t cur_ts_ms = get_time_in_ms();
      float fps = frm_diff * 1000.0 / (cur_ts_ms - last_time_ms);
      CVI_AI_Set_Fall_FPS(pstAIArgs->stAIHandle, fps);  // set only once if fps is stable
      last_time_ms = cur_ts_ms;
      last_counter = counter;
      printf("++++++++++++ frame:%d,fps:%.2f\n", (int)counter, fps);
    }

    //*******************************************
    // Step 1: Object detect inference.
    s32Ret = pstAIArgs->object_detect(pstAIArgs->stAIHandle, &stFrame, &stObjMeta);
    if (s32Ret != CVIAI_SUCCESS) {
      AI_LOGE("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    if (pstAIArgs->bTrackingWithFeature) {
      // Step 2: Extract ReID feature for all person bbox.
      for (uint32_t i = 0; i < stObjMeta.size; i++) {
        if (stObjMeta.info[i].classes == CVI_AI_DET_TYPE_PERSON) {
          s32Ret = CVI_AI_OSNetOne(pstAIArgs->stAIHandle, &stFrame, &stObjMeta, (int)i);
          if (s32Ret != CVIAI_SUCCESS) {
            AI_LOGE("inference failed!, ret=%x\n", s32Ret);
            goto inf_error;
          }
        }
      }
    }

    // Step 3: Multi-Object Tracking inference.
    s32Ret = CVI_AI_DeepSORT_Obj(pstAIArgs->stAIHandle, &stObjMeta, &stTrackerMeta,
                                 pstAIArgs->bTrackingWithFeature);
    if (s32Ret != CVIAI_SUCCESS) {
      AI_LOGE("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    // AI_LOGI("person detect: %d\n", stObjMeta.size);
    //*******************************************

    s32Ret = CVI_AI_Fall_Monitor(pstAIArgs->stAIHandle, &stObjMeta);
    if (s32Ret != CVI_SUCCESS) {
      printf("monitor failed with %#x!\n", s32Ret);
      return -1;
    }

    for (uint32_t i = 0; i < stObjMeta.size; i++) {
      if (stObjMeta.info[i].pedestrian_properity->fall) {
        printf("Falling !!!\n");
      }
    }

    {
      MutexAutoLock(ResultMutex, lock);
      CVI_AI_CopyObjectMeta(&stObjMeta, &g_stObjMeta);
      CVI_AI_CopyTrackerMeta(&stTrackerMeta, &g_stTrackerMeta);
    }

  inf_error:
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    CVI_AI_Free(&stObjMeta);
    CVI_AI_Free(&stTrackerMeta);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }

  AI_LOGI("Exit AI thread\n");
  pthread_exit(NULL);
}

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);
  AI_LOGI("handle signal, signo: %d\n", signo);
  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf(
        "\nUsage: %s DET_MODEL_PATH \n\n"
        "\tDET_MODEL_PATH, path to person detection model\n",
        argv[0]);
    return -1;
  }

  CVI_AI_SUPPORTED_MODEL_E enOdModelId = CVI_AI_SUPPORTED_MODEL_YOLOV8POSE;

  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  SAMPLE_AI_MW_CONFIG_S stMWConfig = {0};

  CVI_S32 s32Ret = SAMPLE_AI_Get_VI_Config(&stMWConfig.stViConfig);
  if (s32Ret != CVI_SUCCESS || stMWConfig.stViConfig.s32WorkingViNum <= 0) {
    AI_LOGE("Failed to get senor infomation from ini file (/mnt/data/sensor_cfg.ini).\n");
    return -1;
  }

  // Get VI size
  PIC_SIZE_E enPicSize;
  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stMWConfig.stViConfig.astViInfo[0].stSnsInfo.enSnsType,
                                          &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("Cannot get senor size\n");
    return -1;
  }

  SIZE_S stSensorSize;
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSensorSize);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("Cannot get senor size\n");
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

  // VBPool 2 for AI preprocessing
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_BGR_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = 768;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = 1024;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].bBind = false;

  // Setup VPSS Grp0
  stMWConfig.stVPSSPoolConfig.u32VpssGrpCount = 1;
  stMWConfig.stVPSSPoolConfig.stVpssMode.aenInput[0] = VPSS_INPUT_MEM;
  stMWConfig.stVPSSPoolConfig.stVpssMode.enMode = VPSS_MODE_DUAL;
  stMWConfig.stVPSSPoolConfig.stVpssMode.ViPipe[0] = 0;
  stMWConfig.stVPSSPoolConfig.stVpssMode.aenInput[1] = VPSS_INPUT_ISP;
  stMWConfig.stVPSSPoolConfig.stVpssMode.ViPipe[1] = 0;

  SAMPLE_AI_VPSS_CONFIG_S *pstVpssConfig = &stMWConfig.stVPSSPoolConfig.astVpssConfig[0];
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
  SAMPLE_AI_Get_Input_Config(&stMWConfig.stVencConfig.stChnInputCfg);
  stMWConfig.stVencConfig.u32FrameWidth = stVencSize.u32Width;
  stMWConfig.stVencConfig.u32FrameHeight = stVencSize.u32Height;

  // Get default RTSP configurations
  SAMPLE_AI_Get_RTSP_Config(&stMWConfig.stRTSPConfig.stRTSPConfig);

  SAMPLE_AI_MW_CONTEXT stMWContext = {0};
  s32Ret = SAMPLE_AI_Init_WM(&stMWConfig, &stMWContext);
  if (s32Ret != CVI_SUCCESS) {
    AI_LOGE("init middleware failed! ret=%x\n", s32Ret);
    return -1;
  }

  cviai_handle_t stAIHandle = NULL;

  // Create AI handle and assign VPSS Grp1 Device 0 to AI SDK
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&stAIHandle, 1, 0), s32Ret, create_ai_fail);

  GOTO_IF_FAILED(CVI_AI_SetVBPool(stAIHandle, 0, 2), s32Ret, create_service_fail);

  CVI_AI_SetVpssTimeout(stAIHandle, 1000);

  cviai_service_handle_t stServiceHandle = NULL;
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&stServiceHandle, stAIHandle), s32Ret,
                 create_service_fail);

  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, enOdModelId, argv[1]), s32Ret, setup_ai_fail);

  bool bTrackingWithFeature = false;

  /**
   * We only track person object in this sample. If you want to track other category of object,
   * please add any category you need to CVI_AI_SelectDetectClass. Additionally, person ReID feature
   * is only meaningful if tracked object is belong to person category. Algorithm would not track
   * non-person object with ReID feature even if enable ReID in CVI_AI_DeepSORT_Obj.
   */

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(stAIHandle, true);
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  set_sample_mot_config(&ds_conf);
  CVI_AI_DeepSORT_SetConfig(stAIHandle, &ds_conf, -1, false);

  pthread_t stVencThread, stAIThread;
  SAMPLE_AI_VENC_THREAD_ARG_S venc_args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  SAMPLE_AI_AI_THREAD_ARG_S ai_args = {
      .enOdModelId = enOdModelId,
      .object_detect = CVI_AI_Yolov8_Pose,
      .stAIHandle = stAIHandle,
      .bTrackingWithFeature = bTrackingWithFeature,
  };

  pthread_create(&stVencThread, NULL, run_venc, &venc_args);
  pthread_create(&stAIThread, NULL, run_ai_thread, &ai_args);

  pthread_join(stVencThread, NULL);
  pthread_join(stAIThread, NULL);

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_AI_DestroyHandle(stAIHandle);
create_ai_fail:
  SAMPLE_AI_Destroy_MW(&stMWContext);

  return 0;
}