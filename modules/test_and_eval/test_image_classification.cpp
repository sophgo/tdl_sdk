#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core.hpp"
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sys_utils.hpp"

int main(int argc, char* argv[]) {
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

  printf("start image classification preprocess config \n");
  // setup preprocess
  VpssPreParam p_preprocess_cfg;

  // image preprocess param: mean & std
  float mean[3] = {123.675, 116.28, 103.52};
  float std[3] = {58.395, 57.12, 57.375};

  for (int i = 0; i < 3; i++) {
    p_preprocess_cfg.mean[i] = mean[i] / std[i];
    p_preprocess_cfg.factor[i] = 1.0 / std[i];
  }

  p_preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;
  p_preprocess_cfg.use_quantize_scale = true;
  p_preprocess_cfg.rescale_type = RESCALE_CENTER;
  p_preprocess_cfg.keep_aspect_ratio = true;

  printf("setup image classification param \n");
  ret = CVI_TDL_Set_Image_Cls_Param(tdl_handle, &p_preprocess_cfg);
  printf("image classification set param success!\n");
  if (ret != CVI_SUCCESS) {
    printf("Can not set image classification parameters %#x\n", ret);
    return ret;
  }

  std::string model_path = argv[1];
  std::string str_src_dir = argv[2];

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_IMAGE_CLASSIFICATION,
                          model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x!\n", ret);
    return ret;
  }

  std::cout << "model opened:" << model_path << std::endl;

  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_TDL_ReadImage(str_src_dir.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888);
  std::cout << "CVI_TDL_ReadImage done!\n";

  if (ret != CVI_SUCCESS) {
    std::cout << "Convert out video frame failed with :" << ret << ".file:" << str_src_dir
              << std::endl;
    // continue;
  }

  cvtdl_class_meta_t cls_meta = {0};

  CVI_TDL_Image_Classification(tdl_handle, &fdFrame, &cls_meta);

  for (uint32_t i = 0; i < 5; i++) {
    printf("no %d class %d score: %f \n", i + 1, cls_meta.cls[i], cls_meta.score[i]);
  }

  CVI_VPSS_ReleaseChnFrame(0, 0, &fdFrame);
  CVI_TDL_Free(&cls_meta);
  CVI_TDL_DestroyHandle(tdl_handle);

  return ret;
}