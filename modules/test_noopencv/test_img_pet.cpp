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
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#if defined(CV181X) || defined(ATHENA2)
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif
int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
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

  std::string strf1(argv[2]);

  printf("---------------------openmodel-----------------------------\n");
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION, argv[1]);
  CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION, 0.1);
  if (ret != CVI_SUCCESS) {
    printf("Open model failed with %#x!\n", ret);
    return ret;
  }

  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  VIDEO_FRAME_INFO_S bg;
  IVE_IMAGE_S image1 = CVI_IVE_ReadImage(ive_handle, strf1.c_str(), IVE_IMAGE_TYPE_U8C3_PACKAGE);
#if defined(CV181X) || defined(ATHENA2)
  int imgw = image1.u32Width;
#else
  int imgw = image1.u16Width;
#endif

  if (imgw == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }
  memset(&bg, 0, sizeof(VIDEO_FRAME_INFO_S));
#if defined(CV181X) || defined(ATHENA2)
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
  cvtdl_object_t obj_meta = {0};
  std::cout << &tdl_handle << std::endl;
  std::cout << &bg << std::endl;
  std::cout << &obj_meta << std::endl;

  ret = CVI_TDL_PersonPet_Detection(tdl_handle, &bg, &obj_meta);
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
  CVI_TDL_Free(&obj_meta);

  CVI_SYS_FreeI(ive_handle, &image1);
  CVI_IVE_DestroyHandle(ive_handle);
  CVI_TDL_DestroyHandle(tdl_handle);

  return ret;
}
