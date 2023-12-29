#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"

cvitdl_handle_t tdl_handle = NULL;
cvitdl_eval_handle_t eval_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <deeplabv3 model path> <image root dir> <result root dir>.\n", argv[0]);
    printf("Deeplabv3 model path: Path to deeplabv3 cvimodel.\n");
    printf("Image root dir: Image root directory.\n");
    printf("Result root dir: Root directory to save result file.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_DEEPLABV3, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_Eval_CityscapesInit(eval_handle, argv[2], argv[3]);
  uint32_t num = 0;
  CVI_TDL_Eval_CityscapesGetImageNum(eval_handle, &num);

  for (uint32_t i = 0; i < num; i++) {
    char *img_name;
    CVI_TDL_Eval_CityscapesGetImage(eval_handle, i, &img_name);
    printf("Read: %s\n", img_name);

    VIDEO_FRAME_INFO_S rgb_frame;
    int ret = CVI_TDL_ReadImage(img_handle, img_name, &rgb_frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Failed to read image: %s\n", img_name);
      return ret;
    }

    VIDEO_FRAME_INFO_S label_frame;
    CVI_TDL_DeeplabV3(tdl_handle, &rgb_frame, &label_frame, NULL);

    CVI_TDL_Eval_CityscapesWriteResult(eval_handle, &label_frame, i);

    CVI_VPSS_ReleaseChnFrame(0, 0, &label_frame);
    free(img_name);
    CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  }

  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_SYS_Exit();
}
