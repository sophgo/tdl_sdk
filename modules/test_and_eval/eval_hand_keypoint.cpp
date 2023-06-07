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
#include "sys_utils.hpp"

std::string run_image_hand_keypoint(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle,
                                    std::string model_path) {
  static int model_init = 0;
  CVI_S32 ret;
  if (model_init == 0) {
    std::cout << "to init hand model\t";

    ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT, model_path.c_str());
    if (ret != CVI_SUCCESS) {
      std::cout << "open model failed:" << model_path << std::endl;
      return "";
    }
    std::cout << "init model done\t";
    model_init = 1;
  }

  cvai_handpose21_meta_ts hand_obj = {0};
  memset(&hand_obj, 0, sizeof(cvai_handpose21_meta_ts));
  hand_obj.size = 1;
  hand_obj.info = (cvai_handpose21_meta_t *)malloc(sizeof(cvai_handpose21_meta_t) * hand_obj.size);
  hand_obj.height = p_frame->stVFrame.u32Height;
  hand_obj.width = p_frame->stVFrame.u32Width;
  for (uint32_t i = 0; i < hand_obj.size; i++) {
    hand_obj.info[i].bbox_x = 0;
    hand_obj.info[i].bbox_y = 0;
    hand_obj.info[i].bbox_w = p_frame->stVFrame.u32Width - 1;
    hand_obj.info[i].bbox_h = p_frame->stVFrame.u32Height - 1;
  }
  ret = CVI_AI_HandKeypoint(ai_handle, p_frame, &hand_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "keypoint hand failed:" << ret << std::endl;
  }
  std::cout << "keypoint hand success" << std::endl;
  // generate detection result
  std::stringstream ss;
  // hand_obj.size
  for (uint32_t i = 0; i < 21; i++) {
    ss << hand_obj.info[0].xn[i] << " " << hand_obj.info[0].yn[i] << "\n";
  }
  CVI_AI_Free(&hand_obj);
  return ss.str();
}

int main(int argc, char *argv[]) {
  std::string model_path(argv[1]);
  std::string image_root(argv[2]);
  std::string image_list(argv[3]);
  std::string dst_root(argv[4]);

  if (image_root.at(image_root.size() - 1) != '/') {
    image_root = image_root + std::string("/");
  }
  if (dst_root.at(dst_root.size() - 1) != '/') {
    dst_root = dst_root + std::string("/");
  };
  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::cout << "to read imagelist:" << image_list << std::endl;
  std::vector<std::string> image_files = read_file_lines(image_list);
  if (image_root.size() == 0) {
    std::cout << ", imageroot empty\n";
    return -1;
  }

  for (size_t i = 0; i < image_files.size(); i++) {
    std::cout << "processing :" << i << "/" << image_files.size() << "\t" << image_files[i]
              << std::endl;
    std::string strf = image_root + image_files[i];
    std::string dstf = dst_root + replace_file_ext(image_files[i], "txt");
    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_AI_ReadImage(strf.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
    std::cout << "CVI_AI_ReadImage done\t";

    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << strf << std::endl;
      continue;
    }

    std::string str_res = run_image_hand_keypoint(&fdFrame, ai_handle, model_path);

    std::cout << "process_funcs done\t";
    std::cout << "str_res.size():" << str_res.size() << std::endl;

    if (str_res.size() > 0) {
      std::cout << "writing file:" << dstf << std::endl;
      FILE *fp = fopen(dstf.c_str(), "w");
      fwrite(str_res.c_str(), str_res.size(), 1, fp);
      fclose(fp);
    }
    std::cout << "write results done\t";
    CVI_AI_ReleaseImage(&fdFrame);
    std::cout << "CVI_AI_ReleaseImage done\t" << std::endl;
  }

  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
