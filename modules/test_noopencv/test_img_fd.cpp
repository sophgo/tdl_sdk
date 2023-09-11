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
#include "ive/ive.h"

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

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  VIDEO_FRAME_INFO_S bg;
  IVE_IMAGE_S image1 = CVI_IVE_ReadImage(ive_handle, strf1.c_str(), IVE_IMAGE_TYPE_U8C3_PLANAR);
  int imgw = image1.u16Width;
  if (imgw == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg, false);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  std::string str_res;
  for (int i = 0; i < 1; i++) {
    cvai_face_t obj_meta = {0};
    ret = CVI_AI_ScrFDFace(ai_handle, &bg, &obj_meta);
    std::stringstream ss;
    ss << "boxes=[";
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
         << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
    }
    str_res = ss.str();
    CVI_AI_Free(&obj_meta);
  }
  std::cout << str_res << std::endl;
  // CVI_AI_ReleaseImage(&bg);
  CVI_SYS_FreeI(ive_handle, &image1);
  CVI_IVE_DestroyHandle(ive_handle);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}