#include <cvi_ive.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sample_comm.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sys_utils.hpp"

#ifndef NO_OPENCV
#include <opencv2/opencv.hpp>

std::vector<cv::Scalar> color = {cv::Scalar(51, 153, 255), cv::Scalar(0, 153, 76),
                                 cv::Scalar(255, 215, 0), cv::Scalar(255, 128, 0),
                                 cv::Scalar(0, 255, 0)};

int color_map[17] = {0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 4, 3, 4, 3, 4, 3};
int line_map[19] = {4, 4, 3, 3, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0};
int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
                       {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
                       {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};

void show_keypoints(VIDEO_FRAME_INFO_S *bg, cvtdl_object_t *obj_meta, std::string save_path,
                    float score) {
  bg->stVFrame.pu8VirAddr[0] =
      (CVI_U8 *)CVI_SYS_MmapCache(bg->stVFrame.u64PhyAddr[0], bg->stVFrame.u32Length[0]);

  cv::Mat img_rgb(bg->stVFrame.u32Height, bg->stVFrame.u32Width, CV_8UC3,
                  bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Stride[0]);

  for (uint32_t i = 0; i < obj_meta->size; i++) {
    for (uint32_t j = 0; j < 17; j++) {
      // printf("j:%d\n",j);
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[i] < score) {
        continue;
      }
      int x = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[j];
      int y = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[j];
      cv::circle(img_rgb, cv::Point(x, y), 5, color[color_map[j]], -1);
    }

    for (uint32_t k = 0; k < 19; k++) {
      // printf("k:%d\n",k);

      int kps1 = skeleton[k][0];
      int kps2 = skeleton[k][1];
      if (obj_meta->info[i].pedestrian_properity->pose_17.score[kps1] < score ||
          obj_meta->info[i].pedestrian_properity->pose_17.score[kps2] < score)
        continue;

      int x1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps1];
      int y1 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps1];

      int x2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.x[kps2];
      int y2 = (int)obj_meta->info[i].pedestrian_properity->pose_17.y[kps2];

      cv::line(img_rgb, cv::Point(x1, y1), cv::Point(x2, y2), color[line_map[k]], 2);
    }
  }

  cv::imwrite(save_path, img_rgb);
  CVI_SYS_Munmap((void *)bg->stVFrame.pu8VirAddr[0], bg->stVFrame.u32Length[0]);
}

#endif

void set_sample_mot_config(cvtdl_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.max_unmatched_num = 10;
  ds_conf->ktracker_conf.accreditation_threshold = 10;
  ds_conf->ktracker_conf.P_beta[2] = 0.1;
  ds_conf->ktracker_conf.P_beta[6] = 2.5e-2;
  ds_conf->kfilter_conf.Q_beta[2] = 0.1;
  ds_conf->kfilter_conf.Q_beta[6] = 2.5e-2;
}

int main(int argc, char *argv[]) {
  // argv[1] : yolov8 pose 路径
  // argv[2] : 图片文件夹路径
  // argv[3] : txt文件，包含所有图像名字，一个图片名一行
  // argv[4] : 画上关键点的图片保存路径
  // argv[5] : smooth 类型，0或1，默认0

  CVI_TDL_SUPPORTED_MODEL_E enOdModelId = CVI_TDL_SUPPORTED_MODEL_YOLOV8POSE;

  std::string pose_model(argv[1]);

  std::string str_image_root(argv[2]);
  std::string image_list(argv[3]);
  std::string dst_dir(argv[4]);

  int smooth_type = atoi(argv[5]);

  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);

  cvitdl_handle_t tdl_handle = NULL;

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  // Init DeepSORT
  CVI_TDL_DeepSORT_Init(tdl_handle, true);
  cvtdl_deepsort_config_t ds_conf;
  CVI_TDL_DeepSORT_GetDefaultConfig(&ds_conf);
  set_sample_mot_config(&ds_conf);
  CVI_TDL_DeepSORT_SetConfig(tdl_handle, &ds_conf, -1, false);

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, pose_model.c_str());

  if (ret != CVI_SUCCESS) {
    printf("open CVI_TDL_SUPPORTED_MODEL_YOLOV8POSE failed with %#x!\n", ret);
    return ret;
  }

  std::cout << "to read imagelist:" << image_list << std::endl;
  std::vector<std::string> image_files = read_file_lines(image_list);
  std::cout << "image_files size: " << image_files.size() << std::endl;

  // FILE* fp = fopen("res.txt", "w");
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  for (size_t i = 0; i < image_files.size(); i++) {
    cvtdl_object_t stObjMeta = {0};
    cvtdl_tracker_t stTrackerMeta = {0};

    std::string file_name = image_files[i];

    std::string strf = join_path(str_image_root, file_name);

    std::cout << "processing :" << i + 1 << "/" << image_files.size() << "\t" << file_name << "\t";

    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_TDL_ReadImage(img_handle, strf.c_str(), &fdFrame, PIXEL_FORMAT_BGR_888);
    std::cout << "ret: " << ret << std::endl;

    if (i == 0) {
      SmoothAlgParam smooth_alg_param = CVI_TDL_Get_Smooth_Algparam(tdl_handle);
      smooth_alg_param.image_width = fdFrame.stVFrame.u32Width;
      smooth_alg_param.image_height = fdFrame.stVFrame.u32Height;
      smooth_alg_param.smooth_type = smooth_type;
      printf("image_width: %d, image_height:%d, smooth_type:%d \n", smooth_alg_param.image_width,
             smooth_alg_param.image_height, smooth_type);
      CVI_TDL_Set_Smooth_Algparam(tdl_handle, smooth_alg_param);
    }

    ret =
        CVI_TDL_PoseDetection(tdl_handle, &fdFrame, CVI_TDL_SUPPORTED_MODEL_YOLOV8POSE, &stObjMeta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_Yolov8_Pose failed with %#x!\n", ret);
    }

    ret = CVI_TDL_DeepSORT_Obj(tdl_handle, &stObjMeta, &stTrackerMeta, false);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_DeepSORT_Obj failed with %#x!\n", ret);
    }

    ret = CVI_TDL_Smooth_Keypoints(tdl_handle, &stObjMeta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_Smooth_Keypoints failed with %#x!\n", ret);
    }

    // std::stringstream res_ss;

    // res_ss << i << ":";
    // if (stObjMeta.size > 0){
    //   for (uint32_t j = 0; j < 17; j++) {
    //     res_ss << stObjMeta.info[0].pedestrian_properity->pose_17.x[j] << ",";
    //     res_ss << stObjMeta.info[0].pedestrian_properity->pose_17.y[j] << ";";
    //   }
    // }

    // res_ss << "\n";
    // fwrite(res_ss.str().c_str(), res_ss.str().size(), 1, fp);

#ifndef NO_OPENCV

    std::string save_path = join_path(dst_dir, file_name);
    show_keypoints(&fdFrame, &stObjMeta, save_path, 0.3);

#endif

    CVI_TDL_ReleaseImage(img_handle, &fdFrame);
    CVI_TDL_Free(&stObjMeta);
    CVI_TDL_Free(&stTrackerMeta);
  }
  // fclose(fp);

  printf("to release system\n");

  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
}
