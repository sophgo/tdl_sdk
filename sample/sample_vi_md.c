#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"
#include "sample_comm.h"
#include "vi_vo_utils.h"
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
        "Usage: %s <video output> <threshold> <min_area>.\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\t threshold: threshold for motion detection [0-255] \n"
        "\t min_area: minimal size of object \n",
        argv[0]);
    return CVI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[1]);
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVI_SUCCESS;
  /*
   * VI pipeline:
   *
   *        __________________________________
   *       |           CHN0 (YUV 400)         +------> MD
   * VI -->| GRP 0     CHN1 (YUV 420 PLANER)  +------> VO
   *       |__________________________________|
   *
   *
   */

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
                    voType != 0, PIXEL_FORMAT_YUV_400);

  if (s32Ret != CVI_SUCCESS) {
    printf("Init video process group 0 failed with %d\n", s32Ret);
    return s32Ret;
  }

  cviai_handle_t ai_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&ai_handle, 2);
  if (ret != CVI_SUCCESS) {
    printf("Facelib open failed with %#x!\n", ret);
    return ret;
  }

  uint32_t threshold = atoi(argv[2]);
  double min_area = atof(argv[3]);

  VIDEO_FRAME_INFO_S stMDFrame, stVOFrame;
  cvai_object_t obj_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));

  int count = 0;

  // interval for updating background
  int update_interval = 1000;

  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stMDFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    if ((count % update_interval) == 0) {
      printf("update background, count=%d, update_interval=%d\n", count, update_interval);
      // Update background. For simplicity, we just set new frame directly.
      if (CVI_AI_Set_MotionDetection_Background(ai_handle, &stMDFrame, threshold, min_area) !=
          CVI_SUCCESS) {
        printf("Cannot update background for motion detection\n");
        break;
      }
    }

    // Detect moving objects. All moving objects are store in obj_meta.
    CVI_AI_MotionDetection(ai_handle, &stMDFrame, &obj_meta);
    printf("detect obj: %d\n", obj_meta.size);

    int s32Ret = CVI_SUCCESS;
    s32Ret = CVI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stMDFrame);
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
      CVI_AI_Service_ObjectDrawRect(NULL, &obj_meta, &stVOFrame, false);
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
    CVI_AI_Free(&obj_meta);
    count++;
  }

  CVI_AI_DestroyHandle(ai_handle);
  DestoryOutput(&outputContext);
  // Exit vpss stuffs
  SAMPLE_COMM_VI_UnBind_VPSS(ViPipe, VpssChn, VpssGrp);
  {
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
    abChnEnable[VpssChn] = CVI_TRUE;
    abChnEnable[VpssChnVO] = CVI_TRUE;
    SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);
  }
  SAMPLE_COMM_VI_DestroyVi(&stViConfig);
  SAMPLE_COMM_SYS_Exit();
}