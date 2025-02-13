#define LOG_TAG "SampleOCC"
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
#include <sys/time.h>
#include <unistd.h>
#include "cvi_kit.h"

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
static volatile bool bExit = false;
static int class = 0;
static float score = 0;
cvtdl_bbox_t crop_bbox = {0};
cvtdl_occlusion_meta_t occlusion_meta = {0};


// static float brightness_th = 150;
static float occ_ratio_th = 0.5;
static float laplacian_th = 10;
static int sensitive_th = 30;

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *pstMWContext;
  cvitdl_service_handle_t stServiceHandle;
} SAMPLE_TDL_VENC_THREAD_ARG_S;

static const char *cls_name[] = {"normal", "covered"};

void *run_venc(void *args) {
  printf("Enter encoder thread\n");
  SAMPLE_TDL_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_TDL_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    MutexAutoLock(ResultMutex, lock);
    char *id_num = calloc(64, sizeof(char));
    sprintf(id_num, "cls: %s, score:%f", cls_name[class], score);
    if (class == 0) {
      CVI_TDL_Service_ObjectWriteText(id_num, 50, 50, &stFrame, -1, 0, 0);
    } else {
      CVI_TDL_Service_ObjectWriteText(id_num, 150, 150, &stFrame, -1, -1, 0);
    }

    free(id_num);

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
  printf("Enter AI thread\n");



  VIDEO_FRAME_INFO_S stFrame;

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
    // struct timeval start_time, stop_time;
    // gettimeofday(&start_time, NULL);

    s32Ret = CVI_TDL_Set_Occlusion_Laplacian(&stFrame, &occlusion_meta);
    if (s32Ret != CVI_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }
    // gettimeofday(&stop_time, NULL);
    // printf("cvfilter Time use %f ms\n",
    //       (__get_us(stop_time) - __get_us(start_time)) / 1000);

    MutexAutoLock(ResultMutex, lock);
    score = occlusion_meta.occ_score;

    class = occlusion_meta.occ_class;

  inf_error:
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

  printf("Exit AI thread\n");
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
  if (argc < 5) {
    printf("\nUsage: %s float[x1] [y1] [x2] [y2] laplacian_th occ_ratio_th sensitive_th.\n\n", argv[0]);
    return CVI_FAILURE;
  }

  int model_input_h = 192;
  int model_input_w = 320;

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

  // VBPool 2 for AI preprocessing.
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = 192;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = 320;
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

  cvitdl_handle_t stAIHandle = NULL;

  GOTO_IF_FAILED(CVI_TDL_CreateHandle2(&stAIHandle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_TDL_SetVBPool(stAIHandle, 0, 2), s32Ret, create_service_fail);

  cvitdl_service_handle_t stServiceHandle = NULL;
  GOTO_IF_FAILED(CVI_TDL_Service_CreateHandle(&stServiceHandle, stAIHandle), s32Ret,
                 create_service_fail);

  crop_bbox.x1 = atof(argv[1]);
  crop_bbox.y1 = atof(argv[2]);
  crop_bbox.x2 = atof(argv[3]);
  crop_bbox.y2 = atof(argv[4]);
  if (argc == 8) {
    laplacian_th = atoi(argv[5]);
    occ_ratio_th = atof(argv[6]);
    sensitive_th = atoi(argv[7]);
  }
  occlusion_meta.crop_bbox = crop_bbox;
  occlusion_meta.laplacian_th = laplacian_th;
  occlusion_meta.occ_ratio_th = occ_ratio_th;
  occlusion_meta.sensitive_th = sensitive_th;

  pthread_t stVencThread, stAIThread;
  SAMPLE_TDL_VENC_THREAD_ARG_S args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
  };

  cvtdl_vpssconfig_t vpssConfig;
  VPSS_CHN_DEFAULT_HELPER(&vpssConfig.chn_attr, model_input_w, model_input_h, PIXEL_FORMAT_BGR_888,
                          false);
  CVI_VPSS_SetChnAttr(0, 1, &vpssConfig.chn_attr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    return CVI_FAILURE;
  }

  pthread_create(&stVencThread, NULL, run_venc, &args);
  pthread_create(&stAIThread, NULL, run_tdl_thread, stAIHandle);

  pthread_join(stVencThread, NULL);
  pthread_join(stAIThread, NULL);

  CVI_TDL_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_TDL_DestroyHandle(stAIHandle);
create_ai_fail:
  SAMPLE_TDL_Destroy_MW(&stMWContext);

  return 0;
}