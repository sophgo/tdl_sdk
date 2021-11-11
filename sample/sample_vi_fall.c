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
    return CVIAI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[3]);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1920, .u32Height = 1080};
  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_RGB_888, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  int ret;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), ret, create_service_fail);

  GOTO_IF_FAILED(
      CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, argv[1]), ret,
      setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, argv[2]), ret,
                 setup_ai_fail);

  GOTO_IF_FAILED(
      CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, false),
      ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN,
                                          1, CVI_AI_DET_TYPE_PERSON),
                 ret, setup_ai_fail);

  VIDEO_FRAME_INFO_S fdFrame, stVOFrame;
  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &fdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    // Run inference and print result.
    CVI_AI_MobileDetV2_Pedestrian(ai_handle, &fdFrame, &obj);
    printf("\nPeople found %x ", obj.size);

    if (obj.size > 0) {
      CVI_AI_AlphaPose(ai_handle, &fdFrame, &obj);

      CVI_AI_Fall(ai_handle, &obj);
      if (obj.size > 0 && obj.info[0].pedestrian_properity != NULL) {
        printf("; fall score %d ", obj.info[0].pedestrian_properity->fall);
      }
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &fdFrame);
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

      if (obj.size > 0) {
        if (obj.info[0].pedestrian_properity && obj.info[0].pedestrian_properity->fall) {
          strcpy(obj.info[0].name, "falling");
        } else {
          strcpy(obj.info[0].name, "");
        }
        CVI_AI_Service_ObjectDrawRect(service_handle, &obj, &stVOFrame, true,
                                      CVI_AI_Service_GetDefaultBrush());
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

    CVI_AI_Free(&obj);
  }

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}