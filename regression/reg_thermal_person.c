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
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

uint32_t coco_ids[] = {1};

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <thermal_person_model_path> <root folder> <evaluate json> <result_json>.\n",
           argv[0]);
    return CVIAI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;

  uint32_t vpssgrp_width = 1280;
  uint32_t vpssgrp_height = 720;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_BGR_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_BGR_888, 5);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cviai_handle_t ai_handle;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_THERMALPERSON, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model thermal_person_detecton failed with %#x!\n", ret);
    return ret;
  }

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVIAI_SUCCESS) {
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
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &frame, PIXEL_FORMAT_BGR_888) != CVIAI_SUCCESS) {
      printf("Read image failed.\n");
      break;
    }
    free(filename);
    cvai_object_t obj;
    CVI_AI_ThermalPerson(ai_handle, &frame, &obj);
    for (int j = 0; j < obj.size; j++) {
      obj.info[j].classes = coco_ids[obj.info[j].classes];
    }
    CVI_AI_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_AI_Free(&obj);
    CVI_AI_ReleaseImage(&frame);
  }
  CVI_AI_Eval_CocoEndEval(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}