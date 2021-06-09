#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf(
        "Usage: %s <detection_model_path> <alphapose_model_path> <video output>.\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[3]);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVI_SUCCESS;
  //****************************************************************
  // Init VI, VO, Vpss
  CVI_U32 DevNum = 0;
  VI_PIPE ViPipe = 0;
  VPSS_GRP VpssGrp = 0;
  VPSS_CHN VpssChn = VPSS_CHN0;
  VPSS_CHN VpssChnVO = VPSS_CHN2;
  CVI_S32 GrpWidth = 1920;
  CVI_S32 GrpHeight = 1080;
  SAMPLE_VI_CONFIG_S stViConfig;
  s32Ret = InitVI(&stViConfig, &DevNum);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video input failed with %d\n", s32Ret);
    return s32Ret;
  }
  if (ViPipe >= DevNum) {
    printf("Not enough devices. Found %u, required index %u.\n", DevNum, ViPipe);
    return CVI_FAILURE;
  }

  const CVI_U32 voWidth = 1280;
  const CVI_U32 voHeight = 720;
  OutputContext outputContext = {0};
  if (voType) {
    OutputType outputType = voType == 1 ? OUTPUT_TYPE_PANEL : OUTPUT_TYPE_RTSP;
    s32Ret = InitOutput(outputType, voWidth, voHeight, &outputContext);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
      return s32Ret;
    }
  }

  s32Ret = InitVPSS_RGB(VpssGrp, VpssChn, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe,
                        voType != 0);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }
  // Init end
  //****************************************************************

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&ai_handle, 1);
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  // Setup model path and model config.
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, false);
  CVI_AI_SelectDetectClass(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1, 1,
                           CVI_AI_DET_TYPE_PERSON);
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Set model alphapose failed with %#x!\n", ret);
    return ret;
  }

  VIDEO_FRAME_INFO_S fdFrame, stVOFrame;
  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &fdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    // Run inference and print result.
    CVI_AI_MobileDetV2_D1(ai_handle, &fdFrame, &obj);
    printf("\nPeople found %x ", obj.size);

    if (obj.size > 0) {
      CVI_AI_AlphaPose(ai_handle, &fdFrame, &obj);

      CVI_AI_Fall(ai_handle, &obj);
      if (obj.size > 0 && obj.info[0].pedestrian_properity != NULL) {
        printf("; fall score %d ", obj.info[0].pedestrian_properity->fall);
      }
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &fdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }

      if (obj.size > 0) {
        if (obj.info[0].pedestrian_properity && obj.info[0].pedestrian_properity->fall) {
          strcpy(obj.info[0].name, "falling");
        } else {
          strcpy(obj.info[0].name, "");
        }
        CVI_AI_Service_ObjectDrawRect(NULL, &obj, &stVOFrame, true);
      }

      s32Ret = SendOutputFrame(&stVOFrame, &outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }
      s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChnVO, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&obj);
  }

  CVI_AI_DestroyHandle(ai_handle);
  DestoryOutput(&outputContext);
  // Exit vpss stuffs
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  abChnEnable[VpssChn] = CVI_TRUE;
  abChnEnable[VpssChnVO] = CVI_TRUE;
  SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}