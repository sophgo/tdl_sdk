#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1280;
static CVI_S32 vpssgrp_height = 720;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_face_path> <dataset dir path> <result dir path>.\n", argv[0]);
    printf("dataset dir path: Wider face validation folder. eg. /mnt/data/WIDER_val\n");
    printf("result dir path: Result directory path. eg. /mnt/data/wider_result\n");
    printf("Using wider face matlab code to evaluate AUC!!\n");
    return CVI_TDL_FAILURE;
  }

  CVI_TDL_PerfettoInit();
  CVI_S32 ret = CVI_TDL_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&facelib_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_TDL_SetModelThreshold(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, 0.005);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  CVI_TDL_Eval_WiderFaceInit(eval_handle, argv[2], argv[3], &imageNum);
  for (uint32_t i = 0; i < imageNum; i++) {
    char *filepath = NULL;
    CVI_TDL_Eval_WiderFaceGetImagePath(eval_handle, i, &filepath);
    VIDEO_FRAME_INFO_S frame;
    cvtdl_face_t face;
    memset(&face, 0, sizeof(cvtdl_face_t));

    CVI_S32 ret = CVI_TDL_ReadImage(img_handle, filepath, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed. %s!\n", filepath);
      continue;
    }
    printf("Run image %s\n", filepath);
    CVI_TDL_RetinaFace(facelib_handle, &frame, &face);
    CVI_TDL_Eval_WiderFaceResultSave2File(eval_handle, i, &frame, &face);
    CVI_TDL_ReleaseImage(img_handle, &frame);
    CVI_TDL_Free(&face);
  }
  CVI_TDL_Eval_WiderFaceClearInput(eval_handle);

  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
