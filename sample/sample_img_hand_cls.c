#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 3) {
    printf("Usage: %s <hand classification model path> <input image path>\n", argv[0]);
    printf("hand classification model path: Path to hand classification model cvimodel.\n");
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

  VIDEO_FRAME_INFO_S bg;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  printf("---------------------read_image-----------------------\n");
  ret = CVI_TDL_ReadImage(img_handle, argv[2], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
    printf("image read,hidth:%d\n", bg.stVFrame.u32Height);
  }

  printf("---------------------openmodel-----------------------\n");
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_HANDCLASSIFICATION, argv[1]);
  CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_HANDCLASSIFICATION, 0.1);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  cvtdl_object_t obj_meta = {0};
  obj_meta.size = 1;
  obj_meta.info = (cvtdl_object_info_t *)malloc(sizeof(cvtdl_object_info_t) * obj_meta.size);
  memset(obj_meta.info, 0, sizeof(cvtdl_object_info_t));

  obj_meta.height = bg.stVFrame.u32Height;
  obj_meta.width = bg.stVFrame.u32Width;

  for (uint32_t i = 0; i < obj_meta.size; i++) {
    obj_meta.info[i].bbox.x1 = 0;
    obj_meta.info[i].bbox.x2 = bg.stVFrame.u32Width - 1;
    obj_meta.info[i].bbox.y1 = 0;
    obj_meta.info[i].bbox.y2 = bg.stVFrame.u32Height - 1;
    obj_meta.info[i].classes = 0;
    printf("init objbox:%f,%f,%f,%f\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
  }
  CVI_TDL_HandClassification(tdl_handle, &bg, &obj_meta);

  if (strlen(obj_meta.info[0].name) > 0) {
    printf("cls result: %s\n", obj_meta.info[0].name);
  } else {
    printf("cls result: (空)\n");
  }
  CVI_TDL_Free(&obj_meta);
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}