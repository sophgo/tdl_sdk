#include <dirent.h>
#include <errno.h>
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

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_e);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

#define CREATE_WRAPPER(realfunc)                                                     \
  int inference_wrapper_##realfunc(cviai_handle_t handle, VIDEO_FRAME_INFO_S *frame, \
                                   cvai_object_t *objects, cvai_obj_det_type_e e) {  \
    return realfunc(handle, frame, objects);                                         \
  }

#define WRAPPER(realfunc) inference_wrapper_##realfunc
CREATE_WRAPPER(CVI_AI_MobileDetV2_Vehicle_D0)
CREATE_WRAPPER(CVI_AI_MobileDetV2_Pedestrian_D0)

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;
  if (strcmp(model_name, "mobiledetv2-lite") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_LITE;
    config->inference = CVI_AI_MobileDetV2_Lite;
  } else if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "mobiledetv2-vehicle-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_VEHICLE_D0;
    config->inference = WRAPPER(CVI_AI_MobileDetV2_Vehicle_D0);
  } else if (strcmp(model_name, "mobiledetv2-pedestrian-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN_D0;
    config->inference = WRAPPER(CVI_AI_MobileDetV2_Pedestrian_D0);
  } else {
    ret = CVI_FAILURE;
  }
  return ret;
}

typedef struct _Argument {
  char model_path[1024];
  ModelConfig model_config;
  char regression_output_path[1024];
  char eval_json_path[1024];
  char image_folder_path[1024];
  char model_name[1024];
} Argument;

int check_dir(const char *dirpath) {
  DIR *dir = opendir(dirpath);
  if (dir) {
    closedir(dir);
    return 0;
  }
  return errno;
}

int check_file(const char *filepath) {
  if (access(filepath, F_OK) == 0) {
    return 0;
  }
  return errno;
}

int parse_args(int argc, char *argv[], Argument *args) {
  if (argc != 5 && argc != 6) {
    printf(
        "Usage: %s <mobiledet-model-path> <image-folder> <evaluate-json> <result-json> "
        "[<model-name>].\n"
        "\n"
        "options:\n"
        "\t<mobiledet-model-path>:\tpath to mobiledet cvimodel\n\n"
        "\t<image-folder>:\t\tpath to image folder\n\n"
        "\t<evaluate-json>:\tpath to coco format json file\n\n"
        "\t<result-json>:\t\toutput path\n\n"
        "\t<model-name> (optional):\tdetection model name should be one of {mobiledetv2-lite, "
        "mobiledetv2-d0, "
        "mobiledetv2-d1, "
        "mobiledetv2-d2, "
        "mobiledetv2-vehicle-d0"
        "mobiledetv2-pedestrian-d0}, default: mobiledetv2-d0\n\n",
        argv[0]);
    return CVI_FAILURE;
  }

  if (argc == 6) {
    strcpy(args->model_name, argv[5]);
  } else {
    strcpy(args->model_name, "mobiledetv2-d0");
  }

  if (createModelConfig(args->model_name, &args->model_config) == CVI_FAILURE) {
    printf("unsupported model: %s\n", args->model_name);
    return CVI_FAILURE;
  }

  int err;
  strcpy(args->model_path, argv[1]);
  if ((err = check_file(args->model_path)) != 0) {
    printf("check model fail: %s, errno: %d\n", args->model_path, err);
    return CVI_FAILURE;
  }

  strcpy(args->image_folder_path, argv[2]);
  if ((err = check_dir(args->image_folder_path)) != 0) {
    printf("check image folder fail: %s, errno: %d\n", args->image_folder_path, err);
    return CVI_FAILURE;
  }

  strcpy(args->eval_json_path, argv[3]);
  if ((err = check_file(args->eval_json_path)) != 0) {
    printf("check json fail: %s, errno: %d\n", args->eval_json_path, err);
    return CVI_FAILURE;
  }

  strcpy(args->regression_output_path, argv[4]);
  return CVI_SUCCESS;
}

int main(int argc, char *argv[]) {
  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  Argument args;
  ret = parse_args(argc, argv, &args);
  if (ret != CVI_SUCCESS) {
    return ret;
  }

  printf("-------------------\n");
  printf("model name: %s\n", args.model_name);
  printf("model path: %s\n", args.model_path);
  printf("image folder: %s\n", args.image_folder_path);
  printf("coco validate json file: %s\n", args.eval_json_path);
  printf("output json path: %s\n", args.regression_output_path);
  printf("-------------------\n");

  uint32_t vpssgrp_width = 1280;
  uint32_t vpssgrp_height = 720;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 2, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 2);
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

  ret = CVI_AI_SetModelPath(ai_handle, args.model_config.model_id, args.model_path);
  if (ret != CVI_SUCCESS) {
    printf("Set model yolov3 failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, args.model_config.model_id, false);
  CVI_AI_SetModelThreshold(ai_handle, args.model_config.model_id, 0.05);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_AI_Eval_CocoInit(eval_handle, args.image_folder_path, args.eval_json_path, &image_num);
  CVI_AI_Eval_CocoStartEval(eval_handle, args.regression_output_path);

  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_AI_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);

    printf("[%d/%d] Reading image %s\n", i + 1, image_num, filename);
    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &blk, &frame, PIXEL_FORMAT_RGB_888_PLANAR) != CVI_SUCCESS) {
      printf("Read image failed.\n");
      break;
    }
    free(filename);
    cvai_object_t obj;
    args.model_config.inference(ai_handle, &frame, &obj, CVI_DET_TYPE_ALL);

    for (int j = 0; j < obj.size; j++) {
      obj.info[j].classes = obj.info[j].classes + 1;
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
