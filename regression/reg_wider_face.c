#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cvimath/cvimath.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1280;
static CVI_S32 vpssgrp_height = 720;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <retina_face_path> <dataset dir path> <result dir path>.\n", argv[0]);
    printf("dataset dir path: Wider face validation folder. eg. /mnt/data/WIDER_val\n");
    printf("result dir path: Result directory path. eg. /mnt/data/wider_result\n");
    printf("Using wider face matlab code to evaluate AUC!!\n");
    return CVIAI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetModelThreshold(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, 0.005);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum;
  CVI_AI_Eval_WiderFaceInit(eval_handle, argv[2], argv[3], &imageNum);
  for (uint32_t i = 0; i < imageNum; i++) {
    char *filepath = NULL;
    CVI_AI_Eval_WiderFaceGetImagePath(eval_handle, i, &filepath);
    VIDEO_FRAME_INFO_S frame;
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));

    CVI_S32 ret = CVI_AI_ReadImage(filepath, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVIAI_SUCCESS) {
      printf("Read image failed. %s!\n", filepath);
      continue;
    }
    printf("Run image %s\n", filepath);
    CVI_AI_RetinaFace(facelib_handle, &frame, &face);
    CVI_AI_Eval_WiderFaceResultSave2File(eval_handle, i, &frame, &face);
    CVI_AI_ReleaseImage(&frame);
    CVI_AI_Free(&face);
  }
  CVI_AI_Eval_WiderFaceClearInput(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
