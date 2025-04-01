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
double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

cvitdl_handle_t tdl_handle = NULL;
static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char* argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <yolo_world_v2 model path> <clip model path> <encoder file> <bpe file> <class txt file> <img file> \n",
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

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_CLIP_TEXT, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model retinaface failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLO_WORLD_V2, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model Yolo_World_V2 failed with %#x!\n", ret);
    return ret;
  }


  printf("to read file_list:%s\n", argv[5]);
  int text_file_count = 80;
  // char** text_file_list = read_file_lines(argv[5], &text_file_count);

  // if (text_file_count == 0) {
  //   printf(", file_list empty\n");
  //   return -1;
  // }

  printf("%d\n", text_file_count);


  int32_t** tokens = (int32_t**)malloc(text_file_count * sizeof(int32_t*));
  ret = CVI_TDL_Set_TextPreprocess(argv[3], argv[4], argv[5], tokens, text_file_count);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_Set_TextPreprocess\n");
    return 0;
  }

  cvtdl_clip_feature** clip_text_feats = (cvtdl_clip_feature**)malloc(text_file_count * sizeof(cvtdl_clip_feature*));

  for (size_t i = 0; i < text_file_count; ++i) {
      clip_text_feats[i] = (cvtdl_clip_feature*)malloc(sizeof(cvtdl_clip_feature));
  }

  for (int i = 0; i < text_file_count; i++) {
    CVI_U8 buffer[77 * sizeof(int32_t)];
    memcpy(buffer, tokens[i], sizeof(int32_t) * 77);
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = buffer;
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = 77;

    ret = CVI_TDL_Clip_Text_Feature(tdl_handle, &Frame, clip_text_feats[i]);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_OpenClip_Text_Feature\n");
      return 0;
    }

  }

  CVI_TDL_NormTextFeature(clip_text_feats, text_file_count);

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  VIDEO_FRAME_INFO_S bg;

  ret = CVI_TDL_ReadImage(img_handle, argv[6], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_object_t obj_meta = {0};

  CVI_TDL_YoloWorldV2(tdl_handle, &bg, clip_text_feats, &obj_meta);

  printf("objnum: %d\n", obj_meta.size);
  printf("boxes=[");

  uint32_t max_num = obj_meta.size > 10? 10:obj_meta.size;
  for (uint32_t i = 0; i < max_num; i++) {
    printf("[%f,%f,%f,%f,%d,%f],", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2, obj_meta.info[i].classes,
           obj_meta.info[i].bbox.score);
  }
  printf("]\n");


  for(int i = 0; i < text_file_count; i++){
    CVI_TDL_Free(clip_text_feats[i]);
  }
  free(clip_text_feats);

  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_Destroy_ImageProcessor(img_handle);

  CVI_TDL_DestroyHandle(tdl_handle);
  return CVI_SUCCESS;
}
