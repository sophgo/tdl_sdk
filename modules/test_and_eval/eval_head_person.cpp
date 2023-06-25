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
std::string g_model_root;

std::string run_image_headperson_detection(VIDEO_FRAME_INFO_S *p_frame, cviai_handle_t ai_handle,
                                           std::string model_name) {
  static int model_init = 0;
  CVI_S32 ret;
  if (model_init == 0) {
    std::cout << "to init headperson model\t";
    std::string str_hand_model = g_model_root + std::string("/") + model_name;

    ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION,
                           str_hand_model.c_str());
    CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_HEAD_PERSON_DETECTION, 0.01);
    if (ret != CVI_SUCCESS) {
      std::cout << "open model failed:" << str_hand_model << std::endl;
      return "";
    }
    std::cout << "init model done\t";
    model_init = 1;
  }

  cvai_object_t hand_obj = {0};
  memset(&hand_obj, 0, sizeof(cvai_object_t));

  ret = CVI_AI_HeadPerson_Detection(ai_handle, p_frame, &hand_obj);
  if (ret != CVI_SUCCESS) {
    std::cout << "detect headperson failed:" << ret << std::endl;
  }

  // generate detection result
  std::stringstream ss;

  for (uint32_t i = 0; i < hand_obj.size; i++) {
    cvai_bbox_t box = hand_obj.info[i].bbox;
    ss << (hand_obj.info[i].classes) << " " << box.x1 << " " << box.y1 << " " << box.x2 << " "
       << box.y2 << " " << box.score << "\n";
  }
  CVI_AI_Free(&hand_obj);
  return ss.str();
}

int main(int argc, char *argv[]) {
  g_model_root = std::string(argv[1]);
  std::string image_root(argv[2]);
  std::string image_list(argv[3]);
  std::string dst_root(argv[4]);
  std::string process_flag(argv[5]);
  std::string model_name(argv[6]);

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
  std::map<std::string,
           std::function<std::string(VIDEO_FRAME_INFO_S *, cviai_handle_t, std::string)>>
      process_funcs = {{"detect", run_image_headperson_detection}};
  if (process_funcs.count(process_flag) == 0) {
    std::cout << "error flag:" << process_flag << std::endl;
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

    std::string str_res = process_funcs[process_flag](&fdFrame, ai_handle, model_name);

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
