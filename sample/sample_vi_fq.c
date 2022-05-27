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
#include <sys/time.h>
#include <unistd.h>

// #define EXECUTION_TIME
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
  if (argc != 5) {
    printf(
        "Usage: %s <retina_model_path> <quality_model_path> <video output> <video input format>.\n"
        "\t video output, 0: disable, 1: output to panel, 2: output through rtsp\n"
        "\t video input format, 0: RGB888, 1: NV21, 2: YUV420\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 voType = atoi(argv[3]);

  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};
  PIXEL_FORMAT_E aiInputFormat;
  int vi_format = atoi(argv[4]);
  if (vi_format == 0) {
    aiInputFormat = PIXEL_FORMAT_RGB_888;
  } else if (vi_format == 1) {
    aiInputFormat = PIXEL_FORMAT_NV21;
  } else if (vi_format == 2) {
    aiInputFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  } else {
    printf("vi format[%d] unknown.\n", vi_format);
    return CVI_FAILURE;
  }
  if (InitVideoSystem(&vs_ctx, &aiInputSize, aiInputFormat, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;

  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]), s32Ret,
                 setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, argv[2]), s32Ret,
                 setup_ai_fail);

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));
  while (bExit == false) {
    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    CVI_AI_RetinaFace(ai_handle, &stfdFrame, &face);
    CVI_AI_Service_FaceAngleForAll(&face);
#ifdef EXECUTION_TIME
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
#endif
    CVI_AI_FaceQuality(ai_handle, &stfdFrame, &face, NULL);
#ifdef EXECUTION_TIME
    gettimeofday(&t1, NULL);
    unsigned long execution_time = ((t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec);
    printf("CVI_AI_FaceQuality execution time: %.2f(ms)\n", (float)execution_time / 1000.);
#endif

    printf("faces number: %u\n", face.size);
    for (uint32_t i = 0; i < face.size; i++) {
      printf("face[%u]: quality[%.4f]\n", i, face.info[i].face_quality);
    }

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
      CVI_AI_Service_FaceDrawRect(service_handle, &face, &stVOFrame, true,
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

    CVI_AI_Free(&face);
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