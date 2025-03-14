#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

// set preprocess and algorithm param for yolov7 detection
// if use official model, no need to change param
CVI_S32 init_param(const cvitdl_handle_t tdl_handle) {
  // setup preprocess
  InputPreParam preprocess_cfg = CVI_TDL_GetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7);

  for (int i = 0; i < 3; i++) {
    printf("asign val %d \n", i);
    preprocess_cfg.factor[i] = 0.003922;
    preprocess_cfg.mean[i] = 0.0;
  }
  preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

  printf("setup yolov7 param \n");
  CVI_S32 ret = CVI_TDL_SetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, preprocess_cfg);
  if (ret != CVI_SUCCESS) {
    printf("Can not set Yolov5 preprocess parameters %#x\n", ret);
    return ret;
  }

  // setup yolo algorithm preprocess
  cvtdl_det_algo_param_t yolov7_param =
      CVI_TDL_GetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7);
  uint32_t *anchors = (uint32_t *)malloc(18 * sizeof(uint32_t));
  uint32_t p_anchors[18] = {12, 16, 19,  36,  40,  28,  36,  75,  76,
                            55, 72, 146, 142, 110, 192, 243, 459, 401};
  memcpy(anchors, p_anchors, sizeof(p_anchors));
  yolov7_param.anchors = anchors;

  uint32_t *strides = (uint32_t *)malloc(3 * sizeof(uint32_t));
  uint32_t p_strides[3] = {8, 16, 32};
  memcpy(strides, p_strides, sizeof(p_strides));
  yolov7_param.strides = strides;
  yolov7_param.cls = 80;

  printf("setup yolov7 algorithm param \n");
  ret = CVI_TDL_SetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, yolov7_param);
  if (ret != CVI_SUCCESS) {
    printf("Can not set Yolov5 algorithm parameters %#x\n", ret);
    return ret;
  }

  // set thershold
  CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, 0.5);
  CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, 0.5);

  printf("yolov7 algorithm parameters setup success!\n");
  return ret;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 3) {
    printf("Usage: %s <yolov7 model path> <input image path>\n", argv[0]);
    printf("yolov7 model path: Path to yolov7 model cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  // change param of ppyoloe
  ret = init_param(tdl_handle);

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x!\n", ret);
    return ret;
  }

  // set thershold
  CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, 0.4);
  CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV7, 0.5);

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_TDL_ReadImage(img_handle, argv[2], &fdFrame, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", fdFrame.stVFrame.u32Width);
    printf("image read,hidth:%d\n", fdFrame.stVFrame.u32Height);
  }

  cvtdl_object_t obj_meta = {0};

  CVI_TDL_Detection(tdl_handle, &fdFrame, CVI_TDL_SUPPORTED_MODEL_YOLOV7, &obj_meta);

  printf("objnum: %d\n", obj_meta.size);
  printf("boxes=[");
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f,%d,%f],", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2, obj_meta.info[i].classes,
           obj_meta.info[i].bbox.score);
  }
  printf("]\n");

  CVI_TDL_ReleaseImage(img_handle, &fdFrame);
  CVI_TDL_Free(&obj_meta);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}