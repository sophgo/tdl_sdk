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

void run_hand_detction(cviai_handle_t ai_handle, std::string model_path,
                       VIDEO_FRAME_INFO_S *p_frame, cvai_object_t &obj_meta) {
  printf("---------------------to do detection-----------------------\n");
  CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_DETECTION, model_path.c_str());
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_DETECTION, 0.5);
  CVI_AI_Hand_Detection(ai_handle, p_frame, &obj_meta);
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    std::cout << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
              << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << ","
              << obj_meta.info[i].classes << "," << obj_meta.info[i].bbox.score << "]" << std::endl;
  }
}

void run_hand_keypoint(cviai_handle_t ai_handle, std::string model_path,
                       VIDEO_FRAME_INFO_S *p_frame, cvai_handpose21_meta_ts &hand_obj,
                       cvai_object_t &obj_meta) {
  printf("---------------------to do detection keypoint-----------------------\n");
  CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT, model_path.c_str());
  hand_obj.size = obj_meta.size;
  hand_obj.info = (cvai_handpose21_meta_t *)malloc(sizeof(cvai_handpose21_meta_t) * hand_obj.size);
  hand_obj.height = p_frame->stVFrame.u32Height;
  hand_obj.width = p_frame->stVFrame.u32Width;
  for (uint32_t i = 0; i < hand_obj.size; i++) {
    hand_obj.info[i].bbox_x = obj_meta.info->bbox.x1;
    hand_obj.info[i].bbox_y = obj_meta.info->bbox.y1;
    hand_obj.info[i].bbox_w = obj_meta.info->bbox.x2 - obj_meta.info->bbox.x1;
    hand_obj.info[i].bbox_h = obj_meta.info->bbox.y2 - obj_meta.info->bbox.y1;
  }
  CVI_AI_HandKeypoint(ai_handle, p_frame, &hand_obj);
  // generate detection result
  for (uint32_t i = 0; i < 21; i++) {
    std::cout << hand_obj.info[0].x[i] << " t  " << hand_obj.info[0].y[i] << "\n";
  }
}

void run_hand_keypoint_cls(cviai_handle_t ai_handle, std::string model_path,
                           cvai_handpose21_meta_ts &hand_obj, cvai_handpose21_meta_t &meta) {
  printf("---------------------to do detection keypoint classification-----------------------\n");
  CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_HAND_KEYPOINT_CLASSIFICATION,
                   model_path.c_str());
  std::vector<float> keypoints;
  for (uint32_t i = 0; i < hand_obj.size; i++) {
    for (uint32_t j = 0; j < 21; j++) {
      keypoints.push_back(hand_obj.info[i].xn[j]);
      keypoints.push_back(hand_obj.info[i].yn[j]);
    }
  }

  if (keypoints.size() != 42) {
    std::cout << "error size " << keypoints.size() << std::endl;
    hand_obj.info[0].label = -1;
    hand_obj.info[0].score = -1.;
  } else {
    CVI_U8 buffer[keypoints.size() * sizeof(float)];
    memcpy(buffer, &keypoints[0], sizeof(float) * keypoints.size());
    VIDEO_FRAME_INFO_S Frame;
    Frame.stVFrame.pu8VirAddr[0] = buffer;  // Global buffer
    Frame.stVFrame.u32Height = 1;
    Frame.stVFrame.u32Width = keypoints.size();
    CVI_AI_HandKeypointClassification(ai_handle, &Frame, &meta);
  }
  printf("meta.label = %d, meta.score = %f\n", meta.label, meta.score);
}

int main(int argc, char *argv[]) {
  std::string hd_model_path(argv[1]);
  std::string kp_model_path(argv[2]);
  std::string kc_model_path(argv[3]);
  std::string image_path(argv[4]);

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

  VIDEO_FRAME_INFO_S fdFrame;
  ret = CVI_AI_ReadImage(image_path.c_str(), &fdFrame, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  }

  cvai_object_t obj_meta = {0};
  cvai_handpose21_meta_ts hand_obj = {0};
  cvai_handpose21_meta_t meta = {0};
  run_hand_detction(ai_handle, hd_model_path, &fdFrame, obj_meta);
  run_hand_keypoint(ai_handle, kp_model_path, &fdFrame, hand_obj, obj_meta);
  run_hand_keypoint_cls(ai_handle, kc_model_path, hand_obj, meta);

  CVI_AI_Free(&obj_meta);
  CVI_AI_Free(&hand_obj);
  CVI_AI_ReleaseImage(&fdFrame);
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
