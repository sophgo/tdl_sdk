#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"
#include "ive/ive.h"

cviai_handle_t ai_handle = NULL;
IVE_HANDLE ive_handle = NULL;
cviai_eval_handle_t eval_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <deeplabv3 model path> <image root dir> <result root dir>.\n", argv[0]);
    printf("Deeplabv3 model path: Path to deeplabv3 cvimodel.\n");
    printf("Image root dir: Image root directory.\n");
    printf("Result root dir: Root directory to save result file.\n");
    return CVIAI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_DEEPLABV3, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  ive_handle = CVI_IVE_CreateHandle();

  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_Eval_CityscapesInit(eval_handle, argv[2], argv[3]);
  uint32_t num = 0;
  CVI_AI_Eval_CityscapesGetImageNum(eval_handle, &num);

  for (uint32_t i = 0; i < num; i++) {
    char *img_name;
    CVI_AI_Eval_CityscapesGetImage(eval_handle, i, &img_name);
    printf("Read: %s\n", img_name);

    IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, img_name, IVE_IMAGE_TYPE_U8C3_PACKAGE);
    if (image.u16Width == 0) {
      printf("Read image failed!\n");
      return CVIAI_FAILURE;
    }

    VIDEO_FRAME_INFO_S rgb_frame;
    CVI_S32 ret = CVI_IVE_Image2VideoFrameInfo(&image, &rgb_frame, false);
    if (ret != CVIAI_SUCCESS) {
      printf("Convert to video frame failed with %#x!\n", ret);
      return ret;
    }

    VIDEO_FRAME_INFO_S label_frame;
    CVI_AI_DeeplabV3(ai_handle, &rgb_frame, &label_frame, NULL);

    CVI_AI_Eval_CityscapesWriteResult(eval_handle, &label_frame, i);

    CVI_SYS_FreeI(ive_handle, &image);
    CVI_VPSS_ReleaseChnFrame(0, 0, &label_frame);
    free(img_name);
  }

  CVI_AI_DestroyHandle(ai_handle);
  CVI_IVE_DestroyHandle(ive_handle);
  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_SYS_Exit();
}
