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
#include <sys/time.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile bool bExit = false;

static char g_answer[64];

static cvtdl_face_t g_stFaceMeta = {0};

MUTEXAUTOLOCK_INIT(ResultMutex);

typedef struct {
  SAMPLE_TDL_MW_CONTEXT *pstMWContext;
  cvitdl_service_handle_t stServiceHandle;
  cvtdl_tokens *tokens;
} SAMPLE_TDL_VENC_THREAD_ARG_S;

typedef struct {
  cvitdl_handle_t stTDLHandle;
  cvtdl_tokens *tokens;
} SAMPLE_TDL_TDL_THREAD_ARG_S;

static uint32_t get_time_in_ms() {
  struct timeval tv;
  if (gettimeofday(&tv, NULL) < 0) {
    return 0;
  }
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void *run_venc(void *args) {
  printf("Enter encoder thread\n");
  SAMPLE_TDL_VENC_THREAD_ARG_S *pstArgs = (SAMPLE_TDL_VENC_THREAD_ARG_S *)args;
  VIDEO_FRAME_INFO_S stFrame;
  CVI_S32 s32Ret;
  cvtdl_face_t stFaceMeta = {0};

  char answer[64];

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, 0, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      MutexAutoLock(ResultMutex, lock);
      strncpy(answer, g_answer, sizeof(answer) - 1); 
    }


    CVI_TDL_Service_ObjectWriteText(answer, 50, 50, &stFrame, 1, 1, 1);

    if (s32Ret != CVI_TDL_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      printf("Draw fame fail!, ret=%x\n", s32Ret);
      goto error;
    }

    s32Ret = SAMPLE_TDL_Send_Frame_RTSP(&stFrame, pstArgs->pstMWContext);
    if (s32Ret != CVI_SUCCESS) {
      CVI_VPSS_ReleaseChnFrame(0, 0, &stFrame);
      printf("Send Output Frame NG, ret=%x\n", s32Ret);
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

void *run_tdl_thread(void *args) {
  printf("Enter TDL thread\n");
  SAMPLE_TDL_TDL_THREAD_ARG_S *pstTDLArgs = (SAMPLE_TDL_TDL_THREAD_ARG_S *)args;


  VIDEO_FRAME_INFO_S stFrame;

  cvtdl_image_embeds embeds_meta = {0};
  cvtdl_tokens tokens_out = {0};

  CVI_S32 s32Ret;
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(0, VPSS_CHN1, &stFrame, 2000);

    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame failed with %#x\n", s32Ret);
      goto get_frame_failed;
    }

    uint32_t start_ms = get_time_in_ms();
    CVI_TDL_Blip_Vqa_Venc(pstTDLArgs->stTDLHandle, &stFrame, &embeds_meta);
    uint32_t venc_ms = get_time_in_ms();

    CVI_TDL_Blip_Vqa_Tenc(pstTDLArgs->stTDLHandle, &embeds_meta, pstTDLArgs->tokens);
    uint32_t tenc_ms = get_time_in_ms();

    CVI_TDL_Blip_Vqa_Tdec(pstTDLArgs->stTDLHandle, &embeds_meta, &tokens_out);
    uint32_t end_ms = get_time_in_ms();

    printf("Infer time: Venc:%dms, Tenc:%dms, Tdec:%dms, total:%dms\n", 
          venc_ms-start_ms, tenc_ms-venc_ms, end_ms-tenc_ms, end_ms-start_ms);


    if (s32Ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", s32Ret);
      goto inf_error;
    }

    s32Ret = CVI_TDL_WordPieceDecode(pstTDLArgs->stTDLHandle, &tokens_out);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_TDL_WordPieceDecode failed with %#x!\n", s32Ret);
      return 0;
    }

    {
      MutexAutoLock(ResultMutex, lock);
      strncpy(g_answer, tokens_out.text[0], sizeof(g_answer) - 1); 
    }


  inf_error:
    CVI_TDL_Free(&embeds_meta);
    CVI_TDL_Free(&tokens_out);
    CVI_VPSS_ReleaseChnFrame(0, 1, &stFrame);
  get_frame_failed:
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
  if (argc != 6) {
    printf(
        "Usage: %s <BLIP_VQA_VENC model path> <BLIP_VQA_TENC model path> <BLIP_VQA_TDEC model path>\n"
        "<vocab file path> <input txt list.txt>\n",
        argv[0]);
    return CVI_TDL_FAILURE;
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
      .u32Width = 1280,
      .u32Height = 720,
  };

  stMWConfig.stVBPoolConfig.u32VBPoolCount = 3;

  // VBPool 0 for VPSS Grp0 Chn0
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32BlkCount = 10;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Height = stSensorSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32Width = stSensorSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssChnBinding = VPSS_CHN0;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[0].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 1 for VPSS Grp0 Chn1
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].enFormat = VI_PIXEL_FORMAT;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32BlkCount = 10;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Height = stVencSize.u32Height;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32Width = stVencSize.u32Width;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].bBind = true;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssChnBinding = VPSS_CHN1;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[1].u32VpssGrpBinding = (VPSS_GRP)0;

  // VBPool 2 for TDL preprocessing
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].enFormat = PIXEL_FORMAT_BGR_888_PLANAR;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32BlkCount = 5;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Height = 720;
  stMWConfig.stVBPoolConfig.astVBPoolSetup[2].u32Width = 1280;
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

  s32Ret = CVI_TDL_WordPieceInit(stTDLHandle, argv[4]);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceInit failed  with %#x!\n", s32Ret);
    return 0;
  }

  cvtdl_tokens tokens = {0};
  s32Ret = CVI_TDL_WordPieceToken(stTDLHandle, argv[5], &tokens);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceToken failed! \n");
    return 0;
  }

  printf("input question:\n");
  printf("======================================\n");
  printf("%s\n", tokens.text[0]);
  printf("======================================\n");

  printf("It will take several minutes to load the blip vqa models, please wait ...\n");
  printf("Warning: rtsp rtsp should be opened after all models are loaded!\n");
  s32Ret = CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_VENC, argv[1]);
  if (s32Ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TENC, argv[2]);
  if (s32Ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  s32Ret = CVI_TDL_OpenModel(stTDLHandle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TDEC, argv[3]);
  if (s32Ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  printf("Load blip vqa models successfully\n");

  pthread_t stVencThread, stTDLThread;
  SAMPLE_TDL_VENC_THREAD_ARG_S args = {
      .pstMWContext = &stMWContext,
      .stServiceHandle = stServiceHandle,
      .tokens = &tokens,
  };

  SAMPLE_TDL_TDL_THREAD_ARG_S ai_args = {
      .stTDLHandle = stTDLHandle,
      .tokens = &tokens,
  };

  pthread_create(&stVencThread, NULL, run_venc, &args);
  pthread_create(&stTDLThread, NULL, run_tdl_thread, &ai_args);

  pthread_join(stVencThread, NULL);
  pthread_join(stTDLThread, NULL);

setup_tdl_fail:
  CVI_TDL_Free(&tokens);
  CVI_TDL_Service_DestroyHandle(stServiceHandle);
create_service_fail:
  CVI_TDL_DestroyHandle(stTDLHandle);
create_tdl_fail:
  SAMPLE_TDL_Destroy_MW(&stMWContext);

	CVI_MSG_Deinit();
  return 0;
}
