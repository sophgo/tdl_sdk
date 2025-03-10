#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include "sys_utils.h"

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 5) {
    printf(
        "Usage: %s <vehicle detection model path> <license plate detection model path> <license "
        "plate recognition model path> <input image path>\n",
        argv[0]);
    printf("vehicle detection model path: Path to vehicle detection model cvimodel.\n");
    printf("license plate detection model path: Path to license plate detection model cvimodel.\n");
    printf(
        "license plate recognition model path: Path to license plate recognition model "
        "cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return CVI_FAILURE;
  }
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

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_WPODNET, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LPRNET_CN, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SelectDetectClass(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, 2,
                            CVI_TDL_DET_TYPE_CAR, CVI_TDL_DET_TYPE_TRUCK);

  VIDEO_FRAME_INFO_S bg;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  ret = CVI_TDL_ReadImage(img_handle, argv[4], &bg, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  cvtdl_object_t obj_meta = {0};

  CVI_TDL_Detection(tdl_handle, &bg, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE, &obj_meta);
  CVI_TDL_LicensePlateDetection(tdl_handle, &bg, &obj_meta);
  if (obj_meta.size > 0) {
    printf("obj_size: %d\n", obj_meta.size);
    for (int i = 0; i < obj_meta.size; i++) {
      printf("obj_meta bbox %f %f %f %f\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
             obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
    }
    ret = CVI_TDL_LicensePlateRecognition_CN(tdl_handle, &bg, &obj_meta);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_LicensePlateRecognition_CN failed with %#x!\n", ret);
      return ret;
    }
    int count = 0;
    for (int i = 0; i < obj_meta.size; i++) {
      if (obj_meta.info[i].vehicle_properity &&
          obj_meta.info[i].vehicle_properity->license_char[0] != '\0') {
        count++;
      }
    }
    printf("Recognition license number = %d\n", count);
    printf("Recognition=[\n");
    for (int i = 0; i < obj_meta.size; i++) {
      if (obj_meta.info[i].vehicle_properity &&
          obj_meta.info[i].vehicle_properity->license_char[0] != '\0') {
        printf("plate %zu; pre License char: %s\n", i,
               obj_meta.info[i].vehicle_properity->license_char);
      }
    }
    printf("]\n");
  } else {
    printf("cannot find license plate\n");
  }
  CVI_TDL_Free(&obj_meta);
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}
