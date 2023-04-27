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

  std::string strf1(argv[2]);
  int eval_perf = 0;
  if (argc > 3) {
    eval_perf = atoi(argv[3]);
  }
  VIDEO_FRAME_INFO_S bg;
  ret = CVI_AI_ReadImage(strf1.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
    printf("image read,hidth:%d\n", bg.stVFrame.u32Height);
  }

  printf("---------------------openmodel-----------------------");
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION, argv[1]);
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION, 0.1);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  printf("---------------------to do detection-----------------------\n");

  std::string str_res;
  cvai_object_t obj_meta = {0};
  CVI_AI_HandFacePerson_Detection(ai_handle, &bg, &obj_meta);

  // CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION, argv[2]);
  // CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_HANDCLASSIFICATION, false);
  // CVI_AI_HandClassification(ai_handle, &bg, &obj_meta);

  std::cout << "objnum:" << obj_meta.size << std::endl;
  std::stringstream ss;
  ss << "boxes=[";
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
       << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << ","
       << obj_meta.info[i].classes << "," << obj_meta.info[i].bbox.score << "],";
  }
  ss << "]\n";
  std::cout << ss.str();
  CVI_AI_Free(&obj_meta);
  if (eval_perf) {
    for (int i = 0; i < 101; i++) {
      cvai_object_t obj_meta = {0};
      CVI_AI_HandFacePerson_Detection(ai_handle, &bg, &obj_meta);
      CVI_AI_Free(&obj_meta);
    }
  }
  CVI_AI_ReleaseImage(&bg);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}
