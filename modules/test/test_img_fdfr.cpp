#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <functional>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include "core/cviai_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "evaluation/cviai_media.h"

static float cal_similarity(cv::Mat feature1, cv::Mat feature2) {
  return feature1.dot(feature2) / (cv::norm(feature1) * cv::norm(feature2));
}

int process_image_file(cviai_handle_t ai_handle, const std::string &imgf, cvai_face_t *p_obj) {
  VIDEO_FRAME_INFO_S bg;

  int ret = CVI_AI_ReadImage(imgf.c_str(), &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    std::cout << "failed to open file:" << imgf << std::endl;
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  ret = CVI_AI_ScrFDFace(ai_handle, &bg, p_obj);
  if (ret != CVI_SUCCESS) {
    printf("CVI_AI_ScrFDFace failed with %#x!\n", ret);
    return ret;
  }

  if (p_obj->size > 0) {
    ret = CVI_AI_FaceRecognition(ai_handle, &bg, p_obj);
    if (ret != CVI_SUCCESS) {
      printf("CVI_AI_FaceAttribute failed with %#x!\n", ret);
      return ret;
    }
  } else {
    printf("cannot find faces\n");
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

  std::string fd_model(argv[1]);  // fd ai_model
  std::string fr_model(argv[2]);  // fr ai_model
  std::string img(argv[3]);       // img1;

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, fd_model.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open CVI_AI_SUPPORTED_MODEL_SCRFDFACE model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, fr_model.c_str());
  if (ret != CVI_SUCCESS) {
    printf("open CVI_AI_SUPPORTED_MODEL_FACERECOGNITION model failed with %#x!\n", ret);
    return ret;
  }

  std::string str_res;
  std::stringstream ss;
  cvai_face_t obj_meta = {0};
  process_image_file(ai_handle, img, &obj_meta);

  ss << "boxes=[";
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
       << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
  }
  str_res = ss.str();
  str_res.at(str_res.length() - 1) = ']';

  std::cout << str_res << std::endl;

  if (argc >= 5) {
    // other picture
    std::string img1(argv[4]);  // img2;
    std::string str_res1;
    std::stringstream ss1;
    cvai_face_t obj_meta1 = {0};
    process_image_file(ai_handle, img1, &obj_meta1);

    ss1 << "boxes=[";
    for (uint32_t i = 0; i < obj_meta1.size; i++) {
      ss1 << "[" << obj_meta1.info[i].bbox.x1 << "," << obj_meta1.info[i].bbox.y1 << ","
          << obj_meta1.info[i].bbox.x2 << "," << obj_meta1.info[i].bbox.y2 << "],";
    }
    str_res1 = ss1.str();
    str_res1.at(str_res1.length() - 1) = ']';
    ;
    std::cout << str_res1 << std::endl;

    // compare cal_similarity
    cvai_feature_t feature = obj_meta.info[0].feature;
    cvai_feature_t feature1 = obj_meta1.info[0].feature;
    cv::Mat mat_feature(feature.size, 1, CV_8SC1);
    cv::Mat mat_feature1(feature1.size, 1, CV_8SC1);
    memcpy(mat_feature.data, feature.ptr, feature.size);
    memcpy(mat_feature1.data, feature1.ptr, feature1.size);
    mat_feature.convertTo(mat_feature, CV_32FC1, 1.);
    mat_feature1.convertTo(mat_feature1, CV_32FC1, 1.);

    float similarity = cal_similarity(mat_feature, mat_feature1);
    std::cout << "similarity:" << similarity << std::endl;
    CVI_AI_Free(&obj_meta1);
  }

  CVI_AI_Free(&obj_meta);
  CVI_AI_DestroyHandle(ai_handle);
  return ret;
}
