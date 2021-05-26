#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

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
  if (argc != 5) {
    printf(
        "Usage: %s <detection_model_name> <detection_model_path> <reid_model_path> <video "
        "output>.\n"
        "\tdetection_model_name: detection model name should be one of {mobiledetv2-d0, "
        "mobiledetv2-d1, mobiledetv2-d2, yolov3}\n"
        "\tdetection_model_path: path to detection model\n"
        "\treid_model_path: path to person reid model\n"
        "\tvideo output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[4]);

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

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t obj_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&ai_handle, 1);
  ret |= CVI_AI_Service_CreateHandle(&obj_handle, ai_handle);
  ret = CVI_AI_SetModelPath(ai_handle, model_config.model_id, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, model_config.model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle);
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.max_distance_iou = 0.8;
  ds_conf.ktracker_conf.feature_budget_size = 10;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf);

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
  CVI_AI_Service_SetIntersect(obj_handle, &pts);

  VIDEO_FRAME_INFO_S stFrame, stVOFrame;
  cvai_object_t obj_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }
    //*******************************************
    // Tracking function calls.
    cvai_area_detect_e *status = NULL;
    // Step 1. Object detect inference.
    model_config.inference(ai_handle, &stFrame, &obj_meta, CVI_DET_TYPE_PEOPLE);
    // Step 2. Object feature generator.
    CVI_AI_OSNet(ai_handle, &stFrame, &obj_meta);
    // Step 3. Tracker.
    CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, true);
    // Step 4. Detect intersection.
    CVI_AI_Service_ObjectDetectIntersect(obj_handle, &stFrame, &obj_meta, &status);
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
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChnVO, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_Service_ObjectDrawRect(NULL, &obj_meta, &stVOFrame, true);
      s32Ret = SendOutputFrame(&stVOFrame, &outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

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

  CVI_AI_Free(&pts);
  CVI_AI_Service_DestroyHandle(obj_handle);
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