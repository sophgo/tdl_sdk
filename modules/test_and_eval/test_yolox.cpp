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
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sys_utils.hpp"

int main(int argc, char* argv[]) {
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
  printf("start yolox preprocess config \n");
  // // setup preprocess
  YoloPreParam p_preprocess_cfg;

  for (int i = 0; i < 3; i++) {
    p_preprocess_cfg.factor[i] = 1.0;
    p_preprocess_cfg.mean[i] = 0.0;
  }
  p_preprocess_cfg.use_quantize_scale = true;
  p_preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

  printf("start yolo algorithm config \n");
  // setup yolo param
  YoloAlgParam p_yolo_param;
  p_yolo_param.cls = 80;

  printf("setup yolox param \n");
  ret = CVI_AI_Set_YOLOX_Param(ai_handle, &p_preprocess_cfg, &p_yolo_param);
  printf("yolox set param success!\n");
  if (ret != CVI_SUCCESS) {
    printf("Can not set YoloX parameters %#x\n", ret);
    return ret;
  }

  std::string model_path = argv[1];
  std::string str_src_dir = argv[2];

  float conf_threshold = 0.5;
  float nms_threshold = 0.5;
  if (argc > 3) {
    conf_threshold = std::stof(argv[3]);
  }

  if (argc > 4) {
    nms_threshold = std::stof(argv[4]);
  }

  printf("start open cvimodel...\n");
  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOX, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x!\n", ret);
    return ret;
  }
  printf("cvimodel open success!\n");
  // set thershold
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOX, conf_threshold);
  CVI_AI_SetModelNmsThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOX, nms_threshold);
  std::cout << "model opened:" << model_path << std::endl;

  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_AI_ReadImage(str_src_dir.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888);
  std::cout << "CVI_AI_ReadImage done!\n";

  if (ret != CVI_SUCCESS) {
    std::cout << "Convert out video frame failed with :" << ret << ".file:" << str_src_dir
              << std::endl;
  }

  cvai_object_t obj_meta = {0};

  CVI_AI_YoloX(ai_handle, &fdFrame, &obj_meta);

  printf("detect number: %d\n", obj_meta.size);
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    printf("detect res: %f %f %f %f %f %d\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2, obj_meta.info[i].bbox.score,
           obj_meta.info[i].classes);
  }

  CVI_VPSS_ReleaseChnFrame(0, 0, &fdFrame);
  CVI_AI_Free(&obj_meta);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}