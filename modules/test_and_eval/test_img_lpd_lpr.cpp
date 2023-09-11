#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"

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

  std::string strf1(argv[3]);

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_LP_DETECTION, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_LP_RECONGNITION, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  VIDEO_FRAME_INFO_S bg;
  ret = CVI_AI_ReadImage(strf1.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_LP_DETECTION, true);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_LP_RECONGNITION, true);
  cvai_object_t obj_meta = {0};
  CVI_AI_License_Plate_Detectionv2(ai_handle, &bg, &obj_meta);
  if (obj_meta.size > 0) {
    printf("obj_size: %d\n", obj_meta.size);
    printf("obj_meta bbox %f %f %f %f\n", obj_meta.info[0].bbox.x1, obj_meta.info[0].bbox.y1,
           obj_meta.info[0].bbox.x2, obj_meta.info[0].bbox.y2);
    ret = CVI_AI_License_Plate_Recognitionv2(ai_handle, &bg, &obj_meta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_AI_License_Plate_Recognitionv2 failed with %#x!\n", ret);
      return ret;
    }
  } else {
    printf("cannot find license plate\n");
  }
  CVI_AI_Free(&obj_meta);
  CVI_AI_ReleaseImage(&bg);
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}