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
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"
#include "sys_utils.hpp"

#define KEYPOINT_DIM 2
#define KEYPOINT_NUM 21
#define KEYPOINT_SIZE KEYPOINT_DIM *KEYPOINT_NUM

std::vector<float> read_keypoint_txt(std::string file_path) {
  std::fstream file(file_path);
  std::vector<float> keypoints;
  if (!file.is_open()) {
    return keypoints;
  }
  std::string line;
  while (getline(file, line)) {
    if (!line.empty()) {
      std::size_t i = line.find(' ');
      try {
        keypoints.push_back(std::stof(line.substr(0, i)));
      } catch (std::out_of_range &) {
        keypoints.push_back(0);
      }
      try {
        keypoints.push_back(std::stof(line.substr(i)));
      } catch (std::out_of_range &) {
        keypoints.push_back(0);
      }
    }
  }
  file.close();
  return keypoints;
}

void test_txt_keypoint_data(const std::string &strf, cviai_handle_t ai_handle,
                            cvai_handpose21_meta_t *meta, CVI_U8 *p_buffer) {
  std::vector<float> keypoints = read_keypoint_txt(strf);

  if (keypoints.size() != 42) {
    std::cout << "error size " << keypoints.size() << std::endl;
    meta->label = -1;
    meta->score = -1;
  } else {
    memcpy(p_buffer, &keypoints[0], sizeof(float) * keypoints.size());
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = p_buffer;  // Global buffer
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = KEYPOINT_SIZE;
    CVI_AI_HandKeypointClassification(ai_handle, &Frame, meta);
  }
}

int main(int argc, char *argv[]) {
  CVI_U8 buffer[KEYPOINT_SIZE * sizeof(float) / sizeof(CVI_U8)];

  cviai_handle_t ai_handle = NULL;
  cvai_handpose21_meta_t meta;
  CVI_S32 ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  std::string model_path = argv[1];
  std::string str_root_dir = argv[2];
  std::string str_list_file = argv[3];
  std::string str_res_file = argv[4];

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION,
                         model_path.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open model failed %#x!\n", ret);
    return ret;
  }
  std::cout << "model opened:" << model_path << std::endl;
  std::vector<std::string> strfiles = read_file_lines(str_list_file);
  FILE *fp = fopen(str_res_file.c_str(), "w");

  size_t num_total = strfiles.size();
  for (size_t i = 0; i < num_total; i++) {
    std::cout << "process:" << i << "/" << num_total << ",file:" << strfiles[i] << std::endl;
    std::string strf = str_root_dir + std::string("/") + strfiles[i];
    test_txt_keypoint_data(strf, ai_handle, &meta, buffer);
    std::string str_res = strfiles[i] + " " + std::to_string(meta.label) + "\n";
    std::cout << str_res;
    fwrite(str_res.c_str(), str_res.length(), 1, fp);
  }
  fclose(fp);
  std::cout << "total:" << strfiles.size() << std::endl;

  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
