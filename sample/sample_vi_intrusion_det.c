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

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVIAI_SUCCESS;

  if (strcmp(model_name, "mobiledetv2-lite") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE;
    config->inference = CVI_AI_MobileDetV2_Lite;
  } else if (strcmp(model_name, "mobiledetv2-lite-person-pets") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE_PERSON_PETS;
    config->inference = CVI_AI_MobileDetV2_Lite_Person_Pets;
  } else if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "mobiledetv2-vehicle-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0;
    config->inference = CVI_AI_MobileDetV2_Vehicle_D0;
  } else if (strcmp(model_name, "mobiledetv2-pedestrian-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0;
    config->inference = CVI_AI_MobileDetV2_Pedestrian_D0;
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
  } else {
    ret = CVIAI_FAILURE;
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
        "Usage: %s <model_name> <model_path> <video output> <threshold>.\n"
        "\t model_name: detection model name should be one of {mobiledetv2-lite, "
        "mobiledetv2-lite-person-pets, "
        "mobiledetv2-d0, "
        "mobiledetv2-d1, "
        "mobiledetv2-d2, "
        "mobiledetv2-vehicle-d0, "
        "mobiledetv2-pedestrian-d0, "
        "yolov3}\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\t threshold (optional): threshold for detection model\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[3]);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  ModelConfig model_config;
  if (createModelConfig(argv[1], &model_config) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1920, .u32Height = 1080};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, VI_PIXEL_FORMAT, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  ret |= CVI_AI_Service_EnableTPUDraw(service_handle, true);

  ret = CVI_AI_SetModelPath(ai_handle, model_config.model_id, argv[2]);
  if (ret != CVIAI_SUCCESS) {
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
    CVI_AI_SetModelThreshold(ai_handle, model_config.model_id, threshold);
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, model_config.model_id, false);

  ret = CVI_AI_SelectDetectClass(ai_handle, model_config.model_id, 1, CVI_AI_DET_TYPE_CUP);

  //========================================
  // setting intrusion detection
  CVI_AI_Service_IntrusionDetect_Init(service_handle);
  /*
      TEST_REGION = [
          [[40, 60], [300,150], [250,300], [120,280], [ 100,120]],
          [[380,200], [480,220], [440,420], [275,380]],
      ]
  */
  printf("[%s:%d]\n", __FILE__, __LINE__);
  float r1[2][5] = {{180, 360, 300, 120, 60}, {60, 150, 400, 480, 120}};
  float r2[2][4] = {{380, 480, 440, 275}, {200, 220, 420, 380}};
  cvai_pts_t test_region_1;
  cvai_pts_t test_region_2;
  test_region_1.size = 5;
  test_region_1.x = malloc(sizeof(float) * test_region_1.size);
  test_region_1.y = malloc(sizeof(float) * test_region_1.size);
  memcpy(test_region_1.x, r1[0], sizeof(float) * test_region_1.size);
  memcpy(test_region_1.y, r1[1], sizeof(float) * test_region_1.size);

  test_region_2.size = 4;
  test_region_2.x = malloc(sizeof(float) * test_region_2.size);
  test_region_2.y = malloc(sizeof(float) * test_region_2.size);
  memcpy(test_region_2.x, r2[0], sizeof(float) * test_region_2.size);
  memcpy(test_region_2.y, r2[1], sizeof(float) * test_region_2.size);

  printf("[%s:%d]\n", __FILE__, __LINE__);
  CVI_AI_Service_IntrusionDetect_SetRegion(service_handle, &test_region_1);
  printf("[%s:%d]\n", __FILE__, __LINE__);
  // CVI_AI_Service_IntrusionDetect_SetRegion(service_handle, &test_region_2);

  //========================================
  cvai_service_brush_t region_brush = CVI_AI_Service_GetDefaultBrush();
  region_brush.color.r = 0;
  region_brush.color.g = 255;
  region_brush.color.b = 255;
  //========================================

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    model_config.inference(ai_handle, &stfdFrame, &obj_meta);
    printf("nums of object %u\n", obj_meta.size);

    int s32Ret = CVI_SUCCESS;
    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_Service_DrawPolygon(service_handle, &stVOFrame, &test_region_1, region_brush);
      CVI_AI_Service_ObjectDrawRect(service_handle, &obj_meta, &stVOFrame, true,
                                    CVI_AI_Service_GetDefaultBrush());
      for (uint32_t i = 0; i < obj_meta.size; i++) {
        bool is_intrusion;
        CVI_AI_Service_IntrusionDetect_BBox(service_handle, &obj_meta.info[i].bbox, &is_intrusion);
        if (is_intrusion) {
          printf("[%u] intrusion! (%.2f,%.2f,%.2f,%.2f)\n", i, obj_meta.info[i].bbox.x1,
                 obj_meta.info[i].bbox.y1, obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
          CVI_AI_Service_ObjectWriteText("Intrusion", obj_meta.info[i].bbox.x1,
                                         obj_meta.info[i].bbox.y1, &stVOFrame, 255, 0, 0);
        }
      }
      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
        break;
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&obj_meta);
  }

  CVI_AI_Free(&test_region_1);
  CVI_AI_Free(&test_region_2);

  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);

  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}