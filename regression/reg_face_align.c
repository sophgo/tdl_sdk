#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cvimath/cvimath.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <fd_path> <img_dir>.\n", argv[0]);
    printf("Fd path: Face detection model path.\n");
    printf("Img dir: Directory of WLFW dataset.\n");
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum = 0;
  CVI_AI_Eval_WflwInit(eval_handle, argv[2], &imageNum);
  for (uint32_t i = 0; i < imageNum; i++) {
    char *name = NULL;
    CVI_AI_Eval_WflwGetImage(eval_handle, i, &name);

    char full_img[1024] = "\0";
    strcat(full_img, argv[2]);
    strcat(full_img, "/imgs/");
    strcat(full_img, name);

    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(full_img, &blk, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));

    CVI_AI_RetinaFace(facelib_handle, &frame, &face);

    printf("img_name: %s\n", full_img);
    if (face.size > 0) {
      CVI_AI_Eval_WflwInsertPoints(eval_handle, i, face.info[0].pts, frame.stVFrame.u32Width,
                                   frame.stVFrame.u32Height);
    }

    free(name);
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk);
  }

  CVI_AI_Eval_WflwDistance(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
