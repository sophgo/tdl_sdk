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

CVI_S32 get_yolo_det(std::string img_path, cviai_handle_t ai_handle, VIDEO_FRAME_INFO_S* fdFrame,
                     cvai_object_t* obj_meta) {
  // printf("reading image file: %s \n", img_path.c_str());
  CVI_S32 ret = CVI_AI_ReadImage(img_path.c_str(), fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
  // printf("frame_width %d \t frame_height %d \n", fdFrame->stVFrame.u32Width,
  //        fdFrame->stVFrame.u32Height);
  if (ret != CVI_SUCCESS) {
    std::cout << "Convert out video frame failed with :" << ret << ".file:" << img_path
              << std::endl;
    // continue;
    return ret;
  }

  CVI_AI_PPYoloE(ai_handle, fdFrame, obj_meta);

  return ret;
}

void bench_mark_all(std::string bench_path, std::string image_root, std::string res_path,
                    cviai_handle_t ai_handle) {
  std::fstream file(bench_path);
  if (!file.is_open()) {
    return;
  }

  std::string line;
  int cnt = 0;
  while (getline(file, line)) {
    if (!line.empty()) {
      stringstream ss(line);
      std::string image_name;
      while (ss >> image_name) {
        cvai_object_t obj_meta = {0};
        VIDEO_FRAME_INFO_S fdFrame;
        if (++cnt % 10 == 0) {
          printf("processing idx: %d\n", cnt);
        }
        CVI_S32 ret = get_yolo_det(image_root + image_name, ai_handle, &fdFrame, &obj_meta);
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
        // std::cout << "write results to file: " << res_path << std::endl;
        std::string save_path = res_path + image_name.substr(0, image_name.length() - 4) + ".txt";
        // printf("save res in path: %s \n", save_path.c_str());
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

  printf("start pp-yoloe preprocess config \n");
  // setup preprocess
  YoloPreParam p_preprocess_cfg;

  float mean[3] = {123.675, 116.28, 103.52};
  float std[3] = {58.395, 57.12, 57.375};

  for (int i = 0; i < 3; i++) {
    p_preprocess_cfg.mean[i] = mean[i] / std[i];
    p_preprocess_cfg.factor[i] = 1.0 / std[i];
  }

  p_preprocess_cfg.use_quantize_scale = true;
  p_preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

  printf("start pp-yoloe algorithm config \n");
  // setup pp-yoloe param
  YoloAlgParam p_yolo_param;
  p_yolo_param.cls = 80;

  printf("setup pp-yoloe param \n");
  ret = CVI_AI_Set_PPYOLOE_Param(ai_handle, &p_preprocess_cfg, &p_yolo_param);
  if (ret != CVI_SUCCESS) {
    printf("Can not set pp-yoloe parameters %#x\n", ret);
    return ret;
  }
  printf("pp-yoloe set param success!\n");

  std::string model_path = argv[1];
  std::string bench_path = argv[2];
  std::string image_root = argv[3];
  std::string res_path = argv[4];

  float conf_threshold = 0.5;
  float nms_threshold = 0.5;
  if (argc > 5) {
    conf_threshold = std::stof(argv[5]);
  }

  if (argc > 6) {
    nms_threshold = std::stof(argv[6]);
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_PPYOLOE, model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x %s!\n", ret, model_path.c_str());
    return ret;
  }
  std::cout << "model opened:" << model_path << std::endl;

  // set thershold
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_PPYOLOE, conf_threshold);
  CVI_AI_SetModelNmsThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_PPYOLOE, nms_threshold);

  printf("set model parameter: conf threshold %f nms_threshold %f \n", conf_threshold,
         nms_threshold);

  bench_mark_all(bench_path, image_root, res_path, ai_handle);

  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}