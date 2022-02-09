#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "sample_utils.h"
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

#include "ive/ive.h"

static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

enum LicenseFormat { taiwan, china };

int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <vehicle_detection_model_namne>\n"
        "          <vehicle_detection_model_path>\n"
        "          <license_plate_detection_model_path>\n"
        "          <license_plate_recognition_model_path>\n"
        "          <license_format (tw/cn)>\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[6]);

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

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);

  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(argv[1], &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, od_model_id, argv[2]), s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, od_model_id, 1, CVI_AI_DET_GROUP_VEHICLE),
                 s32Ret, setup_ai_fail);

  enum LicenseFormat license_format;
  if (strcmp(argv[5], "tw") == 0) {
    license_format = taiwan;
  } else if (strcmp(argv[5], "cn") == 0) {
    license_format = china;
  } else {
    printf("Unknown license type %s\n", argv[5]);
    goto setup_ai_fail;
  }

  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, argv[3]), s32Ret,
                 setup_ai_fail);

  switch (license_format) {
    case taiwan:
      GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_TW, argv[4]), s32Ret,
                     setup_ai_fail);
      break;
    case china:
      GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_CN, argv[4]), s32Ret,
                     setup_ai_fail);
      break;
    default:
      goto setup_ai_fail;
  }

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;

  cvai_object_t vehicle_obj;
  memset(&vehicle_obj, 0, sizeof(cvai_object_t));
  size_t counter = 0;
  while (bExit == false) {
    counter += 1;
    printf("\nFrame: %zu\n", counter);
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    printf("Vehicle Detection ... start\n");
    s32Ret = inference(ai_handle, &stfdFrame, &vehicle_obj);
    printf("Find %u vehicles.\n", vehicle_obj.size);

    /* LP Detection */
    printf("CVI_AI_LicensePlateDetection ... start\n");
    CVI_AI_LicensePlateDetection(ai_handle, &stfdFrame, &vehicle_obj);

    /* LP Recognition */
    printf("CVI_AI_LicensePlateRecognition ... start\n");
    switch (license_format) {
      case taiwan:
        CVI_AI_LicensePlateRecognition_TW(ai_handle, &stfdFrame, &vehicle_obj);
        break;
      case china:
        CVI_AI_LicensePlateRecognition_CN(ai_handle, &stfdFrame, &vehicle_obj);
        break;
      default:
        return CVIAI_FAILURE;
    }

    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        printf("Vec[%zu] ID number: %s\n", i, vehicle_obj.info[i].vehicle_properity->license_char);
      } else {
        printf("Vec[%zu] license plate not found.\n", i);
      }
    }

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
      CVI_AI_Service_ObjectDrawRect(service_handle, &vehicle_obj, &stVOFrame, false,
                                    CVI_AI_Service_GetDefaultBrush());
      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&vehicle_obj);
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
