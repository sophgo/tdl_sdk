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
#ifdef CV181X
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif
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

  printf("---------------------openmodel-----------------------------\n");
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION, argv[1]);
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_PERSON_PETS_DETECTION, 0.1);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  VIDEO_FRAME_INFO_S bg;
  IVE_IMAGE_S image1 = CVI_IVE_ReadImage(ive_handle, strf1.c_str(), IVE_IMAGE_TYPE_U8C3_PACKAGE);
#ifdef CV181X
  int imgw = image1.u32Width;
#else
  int imgw = image1.u16Width;
#endif

  if (imgw == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }
  memset(&bg, 0, sizeof(VIDEO_FRAME_INFO_S));
#ifdef CV181X
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg, false);
#endif

  if (ret != CVI_SUCCESS) {
    printf("Open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("Image read, width:%d\n", bg.stVFrame.u32Width);
  }

  printf("---------------------to do detection-----------------------\n");

  std::string str_res;
  cvai_object_t obj_meta = {0};
  std::cout << &ai_handle << std::endl;
  std::cout << &bg << std::endl;
  std::cout << &obj_meta << std::endl;

  ret = CVI_AI_PersonPet_Detection(ai_handle, &bg, &obj_meta);
  std::cout << ret << std::endl;
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

  CVI_SYS_FreeI(ive_handle, &image1);
  CVI_IVE_DestroyHandle(ive_handle);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}