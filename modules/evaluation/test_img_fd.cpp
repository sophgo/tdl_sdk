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

int process_image_file(cviai_handle_t ai_handle, const std::string &imgf, cvai_face_t *p_obj) {
  VIDEO_FRAME_INFO_S bg;

  int ret = CVI_AI_ReadImage(imgf.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    std::cout << "failed to open file:" << imgf << std::endl;
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }
#ifdef USE_RETINAFACE
  ret = CVI_AI_RetinaFace(ai_handle, &bg, p_obj);
#else
  ret = CVI_AI_ScrFDFace(ai_handle, &bg, p_obj);
#endif
  if (ret != CVI_SUCCESS) {
    printf("process img failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_ReleaseImage(&bg);
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

  std::string strf1(argv[1]);  //"/mnt/data/admin1_data/alios_test/a.jpg");
  std::string modelf(
      "/mnt/data/admin1_data/AI_CV/cv182x/ai_models/scrfd_DW_conv_432_768_int8_2.cvimodel");

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, modelf.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  // VIDEO_FRAME_INFO_S bg;
  // // printf("toread image:%s\n",argv[1]);
  // ret = CVI_AI_ReadImage(strf1.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  // if (ret != CVI_SUCCESS) {
  //   printf("open img failed with %#x!\n", ret);
  //   return ret;
  // } else {
  //   printf("image read,width:%d\n", bg.stVFrame.u32Width);
  // }
  std::string str_res;
  std::string str_res1;

  cvai_face_t obj_meta = {0};
  // CVI_AI_RetinaFace(ai_handle, &bg, &obj_meta);
  // CVI_AI_ScrFDFace(ai_handle, &bg, &obj_meta);
  process_image_file(ai_handle, strf1, &obj_meta);

  std::stringstream ss;
  std::stringstream ss1;
  ss << "boxes=[";
  ss1 << "kpts=[";
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
       << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
    ss1 << "[";
    for (uint32_t j = 0; j < obj_meta.info[i].pts.size; j++) {
      ss1 << obj_meta.info[i].pts.x[j] << "," << obj_meta.info[i].pts.y[j];
      if (j < obj_meta.info[i].pts.size - 1) {
        ss1 << ",";
      }
    }
    ss1 << "],";
  }
  str_res = ss.str();
  str_res1 = ss1.str();
  str_res.at(str_res.length() - 1) = ']';
  str_res1.at(str_res1.length() - 1) = ']';
  CVI_AI_Free(&obj_meta);

  std::cout << str_res << std::endl;
  std::cout << str_res1 << std::endl;

  // CVI_AI_ReleaseImage(&bg);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}
