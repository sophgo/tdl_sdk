#define LOG_TAG "SampleFD"
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

static cvai_face_t g_stFaceMeta = {0};

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_AI_MW_CONTEXT *pstMWContext;
  cviai_service_handle_t stServiceHandle;
} SAMPLE_AI_VENC_THREAD_ARG_S;

void *run_venc(void *args) {
  AI_LOGI("Enter encoder thread\n");
  SAMPLE_AI_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_AI_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvai_face_t stFaceMeta = {0};

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      AI_LOGE("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      MutexAutoLock(ResultMutex, lock);
      CVI_AI_CopyFaceMeta(&g_stFaceMeta, &stFaceMeta);
    }

    // Display face quality score and face angle (roll, pitch, yaw) upon the bbox.
    for (uint32_t u32Fid = 0; u32Fid < stFaceMeta.size; u32Fid++) {
      snprintf(stFaceMeta.info[u32Fid].name, sizeof(stFaceMeta.info[u32Fid].name),
               "FQ: %.2f, R: %.2f, P: %.2f, Y: %.2f", stFaceMeta.info[u32Fid].face_quality,
               stFaceMeta.info[u32Fid].head_pose.roll, stFaceMeta.info[u32Fid].head_pose.pitch,
               stFaceMeta.info[u32Fid].head_pose.yaw);
    }

    s32Ret = CVI_AI_Service_FaceDrawRect(pstArgs->stServiceHandle, &stFaceMeta, &stFrame, true,
                                         CVI_AI_Service_GetDefaultBrush());
    if (s32Ret != CVIAI_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      AI_LOGE("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_AI_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
    if (s32Ret != CVI_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      AI_LOGE("Send Output Frame NG, ret=%x\n", s32Ret);
      goto error;
    }

  error:
    CVI_AI_Free(&stFaceMeta);
    CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      bExit = true;
    }
  }
  AI_LOGI("Exit encoder thread\n");
  pthread_exit(NULL);
}

void *run_ai_thread(void *pHandle) {
  AI_LOGI("Enter AI thread\n");
  cviai_handle_t pstAIHandle = (cviai_handle_t)pHandle;

  VIDEO_FRAME_INFO_S stFrame;
  cvai_face_t stFaceMeta = {0};

  CVI_S32 s32Ret;
  while (bExit == false) {
    GOTO_IF_FAILED(CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000), s32Ret, get_frame_failed);

    // Do face detection to find face bbox and five facial landmarks.
    GOTO_IF_FAILED(CVI_AI_RetinaFace(pstAIHandle, &stFrame, &stFaceMeta), s32Ret, inf_error);

    // Calculate face angle by using five facial landmarks and stores the result to
    // stFaceMeta.info[i].head_pose.
    GOTO_IF_FAILED(CVI_AI_Service_FaceAngleForAll(&stFaceMeta), s32Ret, inf_error);

    struct timeval t0, t1;

    // Predict face quality score.
    gettimeofday(&t0, NULL);
    GOTO_IF_FAILED(CVI_AI_FaceQuality(pstAIHandle, &stFrame, &stFaceMeta, NULL), s32Ret, inf_error);
    gettimeofday(&t1, NULL);
    unsigned long execution_time = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec);

    AI_LOGI("face count: %d, exec time=%lu us\n", stFaceMeta.size, execution_time);
    {
      MutexAutoLock(ResultMutex, lock);
      CVI_AI_CopyFaceMeta(&stFaceMeta, &g_stFaceMeta);
    }

  inf_error:
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
    CVI_AI_Free(&stFaceMeta);
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
  if (argc != 4) {
    printf(
        "\nUsage: %s RETINA_MODEL_PATH QUALITY_MODEL_PATH INPUT_FORMAT\n\n"
        "\tRETINA_MODEL_PATH, path to retinaface model.\n"
        "\tQUALITY_MODEL_PATH, path to face quality model.\n"
        "\tINPUT_FORMAT, input format of face quality model. 0: RGB888, 1: NV21, 2: YUV420.\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

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

  PIXEL_FORMAT_E enInputFormat;
  int vi_format = atoi(argv[3]);
  if (vi_format == 0) {
    enInputFormat = PIXEL_FORMAT_RGB_888;
  } else if (vi_format == 1) {
    enInputFormat = PIXEL_FORMAT_NV21;
  } else if (vi_format == 2) {
    enInputFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  } else {
    printf("vi format[%d] unknown.\n", vi_format);
    return CVI_FAILURE;
  }

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
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].enFormat = enInputFormat;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 3;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 2 for AI preprocessing
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = 720;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = 1280;
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

  // Prepare Vpss Chn1 for AI inference
  VPSS_CHN_DEFAULT_HELPER(&pstVpssConfig->astVpssChnAttr[1], stVencSize.u32Width,
                          stVencSize.u32Height, enInputFormat, true);

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

  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]), s32Ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(stAIHandle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[2]), s32Ret,
                 setup_ai_fail);

  pthread_t stVencThread, stAIThread;
  SAMPLE_AI_VENC_THREAD_ARG_S args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  pthread_create(&stVencThread, NULL, run_venc, &args);
  pthread_create(&stAIThread, NULL, run_ai_thread, stAIHandle);

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