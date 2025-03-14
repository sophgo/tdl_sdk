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

CVI_S32 get_od_model_info(char *model_name, CVI_TDL_SUPPORTED_MODEL_E *model_index) {
  CVI_S32 ret = CVI_SUCCESS;
  if (strcmp(model_name, "yolov5") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV5;
  } else if (strcmp(model_name, "yolov6") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV6;
  } else if (strcmp(model_name, "yolov3") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV3;
  } else if (strcmp(model_name, "yolov7") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV7;
  } else if (strcmp(model_name, "yolov8") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION;
  } else if (strcmp(model_name, "yolox") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOX;
  } else if (strcmp(model_name, "ppyoloe") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_PPYOLOE;
  } else if (strcmp(model_name, "yolov10") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV10_DETECTION;
  } else if (strcmp(model_name, "yolov8-hardhat") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV8_HARDHAT;
  } else if (strcmp(model_name, "fire-smoke") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_YOLOV8_FIRE_SMOKE;
  } else if (strcmp(model_name, "hand") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_HAND_DETECTION;
  } else if (strcmp(model_name, "person-pet") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_PERSON_PETS_DETECTION;
  } else if (strcmp(model_name, "person-vehicle") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION;
  } else if (strcmp(model_name, "hand-face-person") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_HAND_FACE_PERSON_DETECTION;
  } else if (strcmp(model_name, "head-person") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_HEAD_PERSON_DETECTION;
  } else if (strcmp(model_name, "mobiledetv2-coco80") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_COCO80;
  } else if (strcmp(model_name, "mobiledetv2-vehicle") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE;
  } else if (strcmp(model_name, "mobiledetv2-pedestrian") == 0) {
    *model_index = CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN;
  } else {
    ret = CVI_TDL_FAILURE;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 4) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf(
        "model name: detection model name should be one of {mobiledetv2-person-vehicle, "
        "mobiledetv2-person-pets, "
        "mobiledetv2-coco80, "
        "mobiledetv2-vehicle, "
        "mobiledetv2-pedestrian, "
        "yolov3, yolox and so on}.\n");
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
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
  CVI_TDL_SUPPORTED_MODEL_E enOdModelId;
  if (get_od_model_info(argv[1], &enOdModelId) == CVI_TDL_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return -1;
  }
  printf("---------------------openmodel-----------------------\n");
  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  // set theshold
  CVI_TDL_SetModelThreshold(tdl_handle, enOdModelId, 0.5);
  CVI_TDL_SetModelNmsThreshold(tdl_handle, enOdModelId, 0.5);

  printf("---------------------to do detection-----------------\n");

  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  VIDEO_FRAME_INFO_S bg;
  ret = CVI_TDL_ReadImage(img_handle, argv[3], &bg, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
    printf("image read,hidth:%d\n", bg.stVFrame.u32Height);
  }

  cvtdl_object_t obj_meta = {0};
  CVI_TDL_Detection(tdl_handle, &bg, enOdModelId, &obj_meta);

  printf("objnum: %d\n", obj_meta.size);
  printf("boxes=[");
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f,%d,%f],", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2, obj_meta.info[i].classes,
           obj_meta.info[i].bbox.score);
  }
  printf("]\n");

  CVI_TDL_Free(&obj_meta);
  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);

  return ret;
}