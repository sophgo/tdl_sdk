#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

CVI_S32 get_yolov5_det(std::string img_path, cviai_handle_t ai_handle, VIDEO_FRAME_INFO_S* fdFrame,
                       cvai_object_t* obj_meta) {
  printf("reading image file: %s \n", img_path.c_str());
  CVI_S32 ret = CVI_AI_ReadImage(img_path.c_str(), fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
  printf("frame_width %d \t frame_height %d \n", fdFrame->stVFrame.u32Width,
         fdFrame->stVFrame.u32Height);
  if (ret != CVI_SUCCESS) {
    std::cout << "Convert out video frame failed with :" << ret << ".file:" << img_path
              << std::endl;
    // continue;
    return ret;
  }

  CVI_AI_Yolov5(ai_handle, fdFrame, obj_meta);

  return ret;
}

void bench_mark_all(std::string bench_path, std::string image_root, std::string res_path,
                    cviai_handle_t ai_handle) {
  std::fstream file(bench_path);
  if (!file.is_open()) {
    return;
  }

  std::string line;

  while (getline(file, line)) {
    if (!line.empty()) {
      stringstream ss(line);
      std::string image_name;
      while (ss >> image_name) {
        cvai_object_t obj_meta = {0};
        VIDEO_FRAME_INFO_S fdFrame;
        CVI_S32 ret = get_yolov5_det(image_root + image_name, ai_handle, &fdFrame, &obj_meta);
        if (ret != CVI_SUCCESS) {
          CVI_AI_Free(&obj_meta);
          CVI_AI_ReleaseImage(&fdFrame);
          break;
        }
        std::stringstream res_ss;

        for (uint32_t i = 0; i < obj_meta.size; i++) {
          res_ss << obj_meta.info[i].bbox.x1 << " " << obj_meta.info[i].bbox.y1 << " "
                 << obj_meta.info[i].bbox.x2 << " " << obj_meta.info[i].bbox.y2 << " "
                 << obj_meta.info[i].bbox.score << " " << obj_meta.info[i].classes << "\n";
        }
        std::cout << "write results to file: " << res_path << std::endl;
        std::string save_path = res_path + image_name.substr(0, image_name.length() - 4) + ".txt";
        printf("save res in path: %s \n", save_path.c_str());
        FILE* fp = fopen(save_path.c_str(), "w");
        fwrite(res_ss.str().c_str(), res_ss.str().size(), 1, fp);
        fclose(fp);

        CVI_AI_Free(&obj_meta);
        CVI_AI_ReleaseImage(&fdFrame);
        break;
      }
    }
  }

  std::cout << "write done!" << std::endl;
}

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

  printf("start yolov preprocess config \n");
  // setup preprocess
  Yolov5PreParam p_preprocess_cfg;

  for (int i = 0; i < 3; i++) {
    printf("asign val %d \n", i);
    p_preprocess_cfg.factor[i] = 0.003922;
    p_preprocess_cfg.mean[i] = 0.0;
  }
  p_preprocess_cfg.use_quantize_scale = true;
  p_preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

  printf("start yolov algorithm config \n");
  // setup yolov5 param
  YOLOV5AlgParam p_yolov5_param;
  uint32_t p_anchors[3][3][2] = {{{10, 13}, {16, 30}, {33, 23}},
                                 {{30, 61}, {62, 45}, {59, 119}},
                                 {{116, 90}, {156, 198}, {373, 326}}};
  memcpy(p_yolov5_param.anchors, p_anchors, sizeof(*p_anchors) * 12);
  p_yolov5_param.conf_thresh = 0.001;
  p_yolov5_param.nms_thresh = 0.65;

  printf("setup yolov5 param \n");
  ret = CVI_AI_Set_YOLOV5_Param(ai_handle, &p_preprocess_cfg, &p_yolov5_param);
  printf("yolov5 set param success!\n");
  if (ret != CVI_SUCCESS) {
    printf("Can not set Yolov5 parameters %#x\n", ret);
    return ret;
  }

  std::string model_path = argv[1];
  std::string bench_path = argv[2];
  std::string image_root = argv[3];
  std::string res_path = argv[4];

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOV5, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x %s!\n", ret, model_path.c_str());
    return ret;
  }
  std::cout << "model opened:" << model_path << std::endl;

  bench_mark_all(bench_path, image_root, res_path, ai_handle);

  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}