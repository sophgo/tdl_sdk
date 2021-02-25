#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include "cviai_perfetto.h"

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ive/ive.h"

static volatile bool bExit = false;

int main(int argc, char *argv[]) {
  CVI_AI_PerfettoInit();
  if (argc != 6) {
    printf(
        "Usage: %s <vehicle_detection_model_path>\n"
        "\t<use_mobiledet_vehicle (0/1)>\n"
        "\t<license_plate_detection_model_path>\n"
        "\t<license_plate_recognition_model_path>\n"
        "\tvideo output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[5]);

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

  s32Ret = InitVPSS(VpssGrp, VpssChn, VpssChnVO, GrpWidth, GrpHeight, voWidth, voHeight, ViPipe,
                    voType != 0);
  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }
  // Init end
  //****************************************************************

  cviai_handle_t ai_handle = NULL;
  s32Ret = CVI_AI_CreateHandle2(&ai_handle, 1);
  int use_vehicle = atoi(argv[2]);
  if (use_vehicle == 1) {
    printf("set:CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0\n");
    s32Ret |=
        CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0, argv[1]);
  } else if (use_vehicle == 0) {
    printf("set:CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0\n");
    s32Ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, argv[1]);
  } else {
    printf("Unknow det model type.\n");
    return CVI_FAILURE;
  }

  s32Ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_WPODNET, argv[3]);
  s32Ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_LPRNET_TW, argv[4]);
  if (s32Ret != CVI_SUCCESS) {
    printf("open failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;

  cvai_object_t vehicle_obj, license_plate_obj;
  memset(&vehicle_obj, 0, sizeof(cvai_object_t));
  memset(&license_plate_obj, 0, sizeof(cvai_object_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    printf("Vehicle Detection ... start\n");
    if (use_vehicle == 1) {
      CVI_AI_MobileDetV2_Vehicle_D0(ai_handle, &stfdFrame, &vehicle_obj);
    } else {
      CVI_AI_MobileDetV2_D0(ai_handle, &stfdFrame, &vehicle_obj, CVI_DET_TYPE_VEHICLE);
    }
    printf("Find %u vehicles.\n", vehicle_obj.size);

    /* LP Detection */
    printf("CVI_AI_LicensePlateDetection ... start\n");
    CVI_AI_LicensePlateDetection(ai_handle, &stfdFrame, &vehicle_obj);

    /* LP Recognition */
    printf("CVI_AI_LicensePlateRecognition ... start\n");
    CVI_AI_LicensePlateRecognition_TW(ai_handle, &stfdFrame, &vehicle_obj);

    for (size_t i = 0; i < vehicle_obj.size; i++) {
      if (vehicle_obj.info[i].vehicle_properity) {
        printf("Vec[%zu] ID number: %s\n", i, vehicle_obj.info[i].vehicle_properity->license_char);
      } else {
        printf("Vec[%zu] license plate not found.\n", i);
      }
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stfdFrame);
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
      CVI_AI_Service_ObjectDrawRect(&vehicle_obj, &stVOFrame, false);
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

    CVI_AI_Free(&vehicle_obj);
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