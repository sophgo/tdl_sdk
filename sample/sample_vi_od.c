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

static volatile bool bExit = false;

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_e);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;

  if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
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

int main(int argc, char *argv[]) {
  if (argc != 4 && argc != 5) {
    printf(
        "Usage: %s <model_name> <model_path> <open vo 1 or 0> <threshold>.\n"
        "\t model_name: detection model name should be one of {mobiledetv2-d0, mobiledetv2-d1, "
        "mobiledetv2-d2, "
        "yolov3}\n"
        "\t threshold (optional): threshold for detection model\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_BOOL isVoOpened = (strcmp(argv[3], "1") == 0) ? true : false;

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
  CVI_U32 DevNum = 0;
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
  if (isVoOpened) {
    s32Ret = InitVO(voWidth, voHeight, &stVoConfig);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_Init_Video_Output failed with %d\n", s32Ret);
      return s32Ret;
    }
    CVI_VO_HideChn(VoLayer, VoChn);
  }

  s32Ret = InitVPSS(VpssGrp, VpssChn, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe,
                    isVoOpened);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }
  // Init end
  //****************************************************************

  cviai_handle_t facelib_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&facelib_handle, 1);
  ret = CVI_AI_SetModelPath(facelib_handle, model_config.model_id, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }
  if (argc == 5) {
    float threshold = atof(argv[4]);
    if (threshold < 0.0 || threshold > 1.0) {
      printf("wrong threshold value: %f\n", threshold);
      return ret;
    } else {
      printf("set threshold to %f\n", threshold);
    }
    CVI_AI_SetModelThreshold(facelib_handle, model_config.model_id, threshold);
  }
  CVI_AI_SetSkipVpssPreprocess(facelib_handle, model_config.model_id, false);

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stfdFrame, 1000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    model_config.inference(facelib_handle, &stfdFrame, &obj_meta, 0);
    printf("nums of object %u\n", obj_meta.size);

    int s32Ret = CVI_SUCCESS;
    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stfdFrame);
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
      CVI_AI_OBJService_DrawRect(&obj_meta, &stVOFrame, true);
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
  }

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