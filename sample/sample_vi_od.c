#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "sample_utils.h"
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
        "\t model_name: detection model name should be one of {mobiledetv2-person-vehicle, "
        "mobiledetv2-person-pets, "
        "mobiledetv2-coco80, "
        "mobiledetv2-vehicle, "
        "mobiledetv2-pedestrian, "
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

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(argv[1], &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  CVI_S32 s32Ret = CVI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1920, .u32Height = 1080};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, VI_PIXEL_FORMAT, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, od_model_id, argv[2]), s32Ret, create_ai_setup_fail);

  if (argc == 5) {
    float threshold = atof(argv[4]);
    if (threshold < 0.0 || threshold > 1.0) {
      printf("wrong threshold value: %f\n", threshold);
      s32Ret = CVI_FAILURE;
      goto create_ai_setup_fail;
    } else {
      printf("set threshold to %f\n", threshold);
    }
    GOTO_IF_FAILED(CVI_AI_SetModelThreshold(ai_handle, od_model_id, threshold), s32Ret,
                   create_ai_setup_fail);
  }

  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, od_model_id, 3, CVI_AI_DET_TYPE_PERSON,
                                          CVI_AI_DET_TYPE_BANANA, CVI_AI_DET_GROUP_VEHICLE),
                 s32Ret, create_ai_setup_fail);

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

    s32Ret = inference(ai_handle, &stfdFrame, &obj_meta);
    if (s32Ret != CVIAI_SUCCESS) {
      printf("inference fail! ret= %#x\n", s32Ret);
      break;
    }

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
      CVI_AI_Service_ObjectDrawRect(service_handle, &obj_meta, &stVOFrame, true,
                                    CVI_AI_Service_GetDefaultBrush());
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

create_ai_setup_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
  return s32Ret;
}