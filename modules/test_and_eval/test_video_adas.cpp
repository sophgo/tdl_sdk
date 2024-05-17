#include <cvi_ive.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_app/cvi_tdl_app.h"
#include "cvi_tdl_media.h"
#include "sample_comm.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sys_utils.hpp"

#include <opencv2/opencv.hpp>

static const char *enumStr[] = {"NORMAL", "START", "COLLISION_WARNING", "DANGER"};

std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.P_beta[2] = 0.01;
  ds_conf->ktracker_conf.P_beta[6] = 1e-5;

  // ds_conf.kfilter_conf.Q_beta[2] = 0.1;
  ds_conf->kfilter_conf.Q_beta[2] = 0.01;
  ds_conf->kfilter_conf.Q_beta[6] = 1e-5;
  ds_conf->kfilter_conf.R_beta[2] = 0.1;
}

cv::Scalar gen_random_color(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand((uint32_t)seed);

  int r = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  int g = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  int b = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;

  return cv::Scalar(r, g, b);
}

void draw_adas(cvitdl_app_handle_t app_handle, VIDEO_FRAME_INFO_S *bg, std::string save_path) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_Mmap(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  cvtdl_object_t *obj_meta = &app_handle->adas_info->last_objects;
  cvtdl_tracker_t *track_meta = &app_handle->adas_info->last_trackers;
  cvtdl_lane_t *lane_meta = &app_handle->adas_info->lane_meta;

  cv::Scalar box_color;
  for (uint32_t oid = 0; oid < obj_meta->size; oid++) {
    // if (track_meta->info[oid].state == CVI_TRACKER_NEW) {
    //     box_color = cv::Scalar(0, 255, 0);
    // } else if (track_meta->info[oid].state == CVI_TRACKER_UNSTABLE) {
    //     box_color = cv::Scalar(105, 105, 105);
    // } else {  // CVI_TRACKER_STABLE
    //     box_color = gen_random_color(obj_meta->info[oid].unique_id, 64);
    // }
    box_color = gen_random_color(obj_meta->info[oid].unique_id, 64);

    cv::Point top_left((int)obj_meta->info[oid].bbox.x1, (int)obj_meta->info[oid].bbox.y1);
    cv::Point bottom_right((int)obj_meta->info[oid].bbox.x2, (int)obj_meta->info[oid].bbox.y2);

    cv::rectangle(img_rgb, top_left, bottom_right, box_color, 2);

    char txt_info[256];
    snprintf(txt_info, sizeof(txt_info), "S:%d m, V:%.1f m/s, [%s]",
             (int)obj_meta->info[oid].adas_properity.dis, obj_meta->info[oid].adas_properity.speed,
             enumStr[obj_meta->info[oid].adas_properity.state]);

    cv::putText(img_rgb, txt_info, cv::Point(top_left.x, top_left.y - 10), 0, 1, box_color, 2);
  }

  int size = bg->stVFrame.u32Width >= 1080 ? 6 : 3;

  for (int i = 0; i < lane_meta->size; i++) {
    int x0 = (int)lane_meta->lane[i].x[0];
    int y0 = (int)lane_meta->lane[i].y[0];
    int x1 = (int)lane_meta->lane[i].x[1];
    int y1 = (int)lane_meta->lane[i].y[1];

    cv::line(img_rgb, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), size);
  }

  char lane_info[64];

  if (app_handle->adas_info->lane_state == 0) {
    strcpy(lane_info, "NORMAL");
    box_color = cv::Scalar(0, 255, 0);
  } else {
    strcpy(lane_info, "LANE DEPARTURE WARNING !");
    box_color = cv::Scalar(0, 0, 255);
  }

  cv::putText(img_rgb, lane_info,
              cv::Point((int)(0.3 * bg->stVFrame.u32Width), (int)(0.8 * bg->stVFrame.u32Height)), 0,
              size / 3, box_color, size / 3);

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "\nUsage: %s people_vehicle_model_path lane_det_model_path  image_dir_path "
        "output_dir_path img_list_path  \n",
        argv[0]);
    return -1;
  }

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

  uint32_t buffer_size = 20;
  cvitdl_app_handle_t app_handle = NULL;
  ret |= CVI_TDL_APP_CreateHandle(&app_handle, tdl_handle);
  ret |= CVI_TDL_APP_ADAS_Init(app_handle, (uint32_t)buffer_size);

  if (ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    // goto setup_tdl_fail;
  }

  CVI_TDL_DeepSORT_Init(tdl_handle, true);
  cvtdl_deepsort_config_t ds_conf;
  CVI_TDL_DeepSORT_GetDefaultConfig(&ds_conf);
  set_sample_mot_config(&ds_conf);
  CVI_TDL_DeepSORT_SetConfig(tdl_handle, &ds_conf, -1, false);

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LANE_DET, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_TDL_SUPPORTED_MODEL_LANE_DET failed with %#x!\n", ret);
    return ret;
  }

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  std::string str_image_root(argv[3]);
  std::string str_dst_root = std::string(argv[4]);
  std::string image_list(argv[5]);
  // int starti = atoi(argv[5]);
  // int endi = atoi(argv[6]);
  if (!create_directory(str_dst_root)) {
    // std::cout << "create directory:" << str_dst_root << " failed\n";
    std::cout << " \n";
  }
  std::string str_dst_video = join_path(str_dst_root, get_directory_name(str_image_root));
  if (!create_directory(str_dst_video)) {
    // std::cout << "create directory:" << str_dst_video << " failed\n";
    std::cout << " \n";
    // return CVI_FAILURE;
  }

  std::cout << "to read imagelist:" << image_list << std::endl;
  std::vector<std::string> image_files = read_file_lines(image_list);
  if (str_image_root.size() == 0) {
    std::cout << ", imageroot empty\n";
    return -1;
  }
  if (str_image_root.at(str_image_root.size() - 1) != '/') {
    str_image_root = str_image_root + std::string("/");
  }

  for (int img_idx = 0; img_idx < image_files.size(); img_idx++) {
    VIDEO_FRAME_INFO_S bg;

    // std::cout << "processing:" << img_idx << "/1000\n";
    // char szimg[256];
    // sprintf(szimg, "%s/%08d.jpg", str_image_root.c_str(), img_idx);

    std::string strf = str_image_root + image_files[img_idx];

    // std::cout << "processing:" << img_idx << "/1000,path:" << szimg << std::endl;

    std::cout << "processing :" << img_idx + 1 << "/" << image_files.size() << "\t"
              << image_files[img_idx] << std::endl;

    ret = CVI_TDL_ReadImage(img_handle, strf.c_str(), &bg, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("open img failed with %#x!\n", ret);
      return ret;
    } else {
      printf("image read,width:%d\n", bg.stVFrame.u32Width);
    }

    ret = CVI_TDL_APP_ADAS_Run(app_handle, &bg);

    if (ret != CVI_TDL_SUCCESS) {
      printf("inference failed!, ret=%x\n", ret);
      CVI_TDL_APP_DestroyHandle(app_handle);

      // goto inf_error;
    }

    cvtdl_lane_t lane_meta = {0};
    CVI_TDL_Lane_Det(app_handle->tdl_handle, &bg, &lane_meta);

    cvtdl_object_t *obj_meta = &app_handle->adas_info->last_objects;

    // for (uint32_t i = 0; i < obj_meta->size; i++) {
    //     std::cout << "[" << obj_meta->info[i].bbox.x1 << "," << obj_meta->info[i].bbox.y1 << ","
    //         << obj_meta->info[i].bbox.x2 << "," << obj_meta->info[i].bbox.y2 << "]," <<std::endl;
    // }

    char save_path[256];
    sprintf(save_path, "%s/%08d.jpg", str_dst_video.c_str(), img_idx);
    draw_adas(app_handle, &bg, save_path);

    CVI_TDL_ReleaseImage(img_handle, &bg);
    CVI_TDL_Free(&lane_meta);
  }

  CVI_TDL_APP_DestroyHandle(app_handle);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}
