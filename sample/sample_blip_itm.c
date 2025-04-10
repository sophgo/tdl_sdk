#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_kit.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sys_utils.h"

cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf(
        "Usage: %s <blip itm model path> <input image path> <vocab file path> <input txt list.txt>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }


  ret = CVI_TDL_WordPieceInit(tdl_handle, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceInit failed  with %#x!\n", ret);
    return 0;
  }


  cvtdl_tokens tokens = {0};
  ret = CVI_TDL_WordPieceToken(tdl_handle, argv[4], &tokens);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceToken failed! \n");
    return 0;
  }


  printf("It will take several minutes to load the blip itm models, please wait ...\n");

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_ITM, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  printf("input txt:\n");
  printf("======================================\n");
  for(int i = 0; i < tokens.sentences_num; i++){
    printf("%s\n", tokens.text[i]);
  }
  printf("======================================\n");

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  VIDEO_FRAME_INFO_S bg;
  ret = CVI_TDL_ReadImage(img_handle, argv[2], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_class_meta_t cls_meta = {0};

  CVI_TDL_Blip_Itm(tdl_handle, &bg, &tokens, &cls_meta);

  printf("max similarity: %f, index: %d\n", cls_meta.score[0], cls_meta.cls[0]);

  CVI_TDL_Free(&tokens);
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_DestroyHandle(tdl_handle);

  return CVI_SUCCESS;
}
