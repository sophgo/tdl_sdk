#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cvimath/cvimath.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"

uint32_t coco_ids[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
                       18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                       37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
                       54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
                       74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <yolo_model_path> <root folder> <evaluate json> <result_json>.\n", argv[0]);
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  uint32_t vpssgrp_width = 1280;
  uint32_t vpssgrp_height = 720;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOV3, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model yolov3 failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_YOLOV3, 0.005);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_AI_Eval_CocoInit(eval_handle, argv[2], argv[3], &image_num);
  CVI_AI_Eval_CocoStartEval(eval_handle, argv[4]);
  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_AI_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);
    printf("Reading image %s\n", filename);
    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &blk, &frame, PIXEL_FORMAT_RGB_888_PLANAR) != CVI_SUCCESS) {
      printf("Read image failed.\n");
      break;
    }
    free(filename);
    cvai_object_t obj;
    cvai_obj_det_type_e det_type = CVI_DET_TYPE_ALL;
    CVI_AI_Yolov3(ai_handle, &frame, &obj, det_type);
    for (int j = 0; j < obj.size; j++) {
      obj.info[j].classes = coco_ids[obj.info[j].classes];
    }
    CVI_AI_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_AI_Free(&obj);
    CVI_VB_ReleaseBlock(blk);
  }
  CVI_AI_Eval_CocoEndEval(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}