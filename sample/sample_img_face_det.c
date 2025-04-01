#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

CVI_S32 get_fd_model_info(char *model_name, CVI_TDL_SUPPORTED_MODEL_E *model_index) {
  CVI_S32 ret = CVI_SUCCESS;
  if (strcmp(model_name, "scrfdface") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_SCRFDFACE;
  } else if (strcmp(model_name, "retinaface") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_RETINAFACE;
  } else if (strcmp(model_name, "retinaface_ir") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_RETINAFACE_IR;
  } else if (strcmp(model_name, "face_mask") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_FACEMASKDETECTION;
  } else {
    ret = CVI_TDL_FAILURE;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 4) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf(
        "model name: face detection model name should be one of {scrfdface, "
        "retinaface, "
        "retinaface_ir, "
        "face_mask}.\n");
    printf("model path: Path to cvimodel.\n");
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
  CVI_TDL_SUPPORTED_MODEL_E enOdModelId;
  if (get_fd_model_info(argv[1], &enOdModelId) == CVI_TDL_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return -1;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  VIDEO_FRAME_INFO_S bg;
  ret = CVI_TDL_ReadImage(img_handle, argv[3], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_face_t obj_meta = {0};
  ret = CVI_TDL_FaceDetection(tdl_handle, &bg, enOdModelId, &obj_meta);
  printf("boxes=[");
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f],", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
  }
  printf("]\n");
  CVI_TDL_Free(&obj_meta);

  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}
