#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

CVI_S32 init_param(const cvitdl_handle_t tdl_handle) {
  // setup preprocess
  InputPreParam preprocess_cfg =
      CVI_TDL_GetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);

  for (int i = 0; i < 3; i++) {
    printf("asign val %d \n", i);
    preprocess_cfg.factor[i] = 0.003922;
    preprocess_cfg.mean[i] = 0.0;
  }
  preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;
  printf("setup yolov8 param \n");
  CVI_S32 ret =
      CVI_TDL_SetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, preprocess_cfg);
  if (ret != CVI_SUCCESS) {
    printf("Can not set yolov8 preprocess parameters %#x\n", ret);
    return ret;
  }

  // setup yolo algorithm preprocess
  cvtdl_det_algo_param_t yolov8_param =
      CVI_TDL_GetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);
  yolov8_param.cls = 1;

  printf("setup yolov8 algorithm param \n");
  ret = CVI_TDL_SetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION,
                                      yolov8_param);
  if (ret != CVI_SUCCESS) {
    printf("Can not set yolov8 algorithm parameters %#x\n", ret);
    return ret;
  }

  // set theshold
  CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, 0.5);
  CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, 0.5);

  printf("yolov8 algorithm parameters setup success!\n");
  return ret;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 5) {
    printf(
        "Usage: %s <license plate detection model path> <license plate keypoint model path> "
        "<license plate recognition model path> <input image path>\n",
        argv[0]);
    printf("license plate detection model path: Path to license plate detection model cvimodel.\n");
    printf("license plate keypoint model path: Path to license plate keypoint model cvimodel.\n");
    printf(
        "license plate recognition model path: Path to license plate recognition model "
        "cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 4,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 4);
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

  ret = init_param(tdl_handle);
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model DETECTION failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_KEYPOINT, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model KEYPOINT failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LP_RECONGNITION, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("open model RECONGNITION failed with %#x!\n", ret);
    return ret;
  }
  char *strf1 = argv[4];

  VIDEO_FRAME_INFO_S bg;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  ret = CVI_TDL_ReadImage(img_handle, strf1, &bg, PIXEL_FORMAT_BGR_888);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_object_t obj_meta = {0};

  CVI_TDL_Detection(tdl_handle, &bg, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, &obj_meta);
  printf("obj_size: %d\n", obj_meta.size);
  if (obj_meta.size > 0) {
    CVI_TDL_License_Plate_Keypoint(tdl_handle, &bg, &obj_meta);
    CVI_TDL_LicensePlateRecognition(tdl_handle, &bg, CVI_TDL_SUPPORTED_MODEL_LP_RECONGNITION,
                                    &obj_meta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_LicensePlateRecognition failed with %#x!\n", ret);
      return ret;
    }
    for (int i = 0; i < obj_meta.size; i++) {
      printf("obj_meta bbox %f %f %f %f\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
             obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
      // 处理车牌字符串，去掉空格
      char license_str[20];  // 假设车牌字符的最大长度为19
      strncpy(license_str, obj_meta.info[i].vehicle_properity->license_char,
              sizeof(license_str) - 1);
      license_str[sizeof(license_str) - 1] = '\0';  // 确保字符串以 '\0' 结尾

      // 去掉空格
      char *src = license_str;
      char *dst = license_str;
      while (*src) {
        if (*src != ' ') {
          *dst++ = *src;  // 复制非空格字符
        }
        src++;
      }
      *dst = '\0';  // 结束字符串

      printf("plate %zu; pre License char: %s\n", i, license_str);
    }
  } else {
    printf("cannot find license plate\n");
  }
  CVI_TDL_Free(&obj_meta);
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}
