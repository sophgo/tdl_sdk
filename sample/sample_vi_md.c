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
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_YUV_400, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  int ret = CVI_AI_CreateHandle2(&ai_handle, 2, 0);
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
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stMDFrame, 2000);
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
    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stMDFrame);
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
      CVI_AI_Service_ObjectDrawRect(NULL, &obj_meta, &stVOFrame, false);
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
    count++;
  }

  CVI_AI_DestroyHandle(ai_handle);
  DestroyVideoSystem(&vs_ctx);
  SAMPLE_COMM_SYS_Exit();
}