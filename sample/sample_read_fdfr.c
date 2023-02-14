#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
// #include "ive/ive.h"
// #include "sys_utils.hpp"
int ReleaseImage(VIDEO_FRAME_INFO_S *frame) {
  CVI_S32 ret = CVI_SUCCESS;
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    ret = CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}
int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_AI_SUPPORTED_MODEL_SCRFDFACE model failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_AI_SUPPORTED_MODEL_FACERECOGNITION model failed with %#x!\n", ret);
    return ret;
  }
  VIDEO_FRAME_INFO_S bg;

  printf("to read image\n");
  if (CVI_SUCCESS != CVI_AI_LoadBinImage(argv[3], &bg, PIXEL_FORMAT_RGB_888_PLANAR)) {
    printf("cviai read image failed.");
    CVI_AI_DestroyHandle(ai_handle);
    return -1;
  }

  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }
  cvai_face_t obj_meta = {0};
  ret = CVI_AI_ScrFDFace(ai_handle, &bg, &obj_meta);
  if (ret != CVI_SUCCESS) {
    printf("CVI_AI_ScrFDFace failed with %#x!\n", ret);
    return ret;
  }
  for (int i = 0; i < obj_meta.size; i++) {
    printf("box=[%.1f,%.1f,%.1f,%.1f],pts=[", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
    for (int j = 0; j < 5; j++) {
      printf("%.1f,%.1f,", obj_meta.info[i].pts.x[j], obj_meta.info[i].pts.y[j]);
    }
  }
  if (obj_meta.size > 0) {
    ret = CVI_AI_FaceRecognition(ai_handle, &bg, &obj_meta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_AI_FaceAttribute failed with %#x!\n", ret);
      return ret;
    }
  } else {
    printf("cannot find faces\n");
  }
  CVI_AI_Free(&obj_meta);
  ReleaseImage(&bg);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}
