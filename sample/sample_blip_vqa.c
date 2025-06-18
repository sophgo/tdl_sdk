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
  if (argc != 7) {
    printf(
        "Usage: %s <BLIP_VQA_VENC model path> <BLIP_VQA_TENC model path> <BLIP_VQA_TDEC model path>\n"
        "<input image path> <vocab file path> <input txt list.txt>\n",
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

  ret = CVI_TDL_WordPieceInit(tdl_handle, argv[5]);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceInit failed  with %#x!\n", ret);
    return 0;
  }


  cvtdl_tokens tokens_in = {0};
  ret = CVI_TDL_WordPieceToken(tdl_handle, argv[6], &tokens_in);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceToken failed  with %#x!\n", ret);
    return 0;
  }


  printf("It will take several minutes to load the blip vqa models, please wait ...\n");

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_VENC, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TENC, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_BLIP_VQA_TDEC, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  VIDEO_FRAME_INFO_S bg;
  ret = CVI_TDL_ReadImage(img_handle, argv[4], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_image_embeds embeds_meta = {0};
  cvtdl_tokens tokens_out = {0};

  CVI_TDL_Blip_Vqa_Venc(tdl_handle, &bg, &embeds_meta);

  CVI_TDL_Blip_Vqa_Tenc(tdl_handle, &embeds_meta, &tokens_in);

  CVI_TDL_Blip_Vqa_Tdec(tdl_handle, &embeds_meta, &tokens_out);

  ret = CVI_TDL_WordPieceDecode(tdl_handle, &tokens_out);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_WordPieceDecode failed with %#x!\n", ret);
    return 0;
  }

  printf("question: %s\n", tokens_in.text[0]);
  printf("answer: %s\n", tokens_out.text[0]);

  CVI_TDL_Free(&tokens_in);
  CVI_TDL_Free(&tokens_out);
  CVI_TDL_Free(&embeds_meta);

  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_DestroyHandle(tdl_handle);

  return CVI_SUCCESS;
}
