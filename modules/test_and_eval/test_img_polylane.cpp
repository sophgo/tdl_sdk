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
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

void show_points(VIDEO_FRAME_INFO_S *bg, cvtdl_lane_t *lane_meta, std::string save_path) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  int det_num = lane_meta->size;

  for (int i = 0; i < det_num; i++) {
    for (int j = 0; j < 56; j++) {
      int x = (int)lane_meta->lane[i].x[j];
      int y = (int)lane_meta->lane[i].y[j];
      cv::circle(img_rgb, cv::Point(x, y), 7, color[i], -1);
    }
  }

  cv::imwrite(save_path.c_str(), img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 2,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 2);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  std::string strf1(argv[2]);
  std::string save_path(argv[3]);

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_POLYLANE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_TDL_Set_Polylanenet_Lower(tdl_handle, CVI_TDL_SUPPORTED_MODEL_POLYLANE, 0.4);
  if (ret != CVI_SUCCESS) {
    printf("Set_Polylanenet_Lower failed with %#x!\n", ret);
    return ret;
  }
  VIDEO_FRAME_INFO_S bg;
  // printf("toread image:%s\n",argv[1]);
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  ret = CVI_TDL_ReadImage(img_handle, strf1.c_str(), &bg, PIXEL_FORMAT_BGR_888);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }
  // std::string str_res;
  for (int i = 0; i < 1; i++) {
    cvtdl_lane_t lane_meta = {0};
    CVI_TDL_PolyLane_Det(tdl_handle, &bg, &lane_meta);
    show_points(&bg, &lane_meta, save_path);
    printf("obj_size: %d\n", lane_meta.size);
    CVI_TDL_Free(&lane_meta);
  }

  // std::cout << str_res << std::endl;
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}