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
#include "evaluation/cviai_media.h"
#include "od_utils.h"

typedef struct _Argument {
  char model_path[1024];
  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
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
        "\t<model-name> (optional):\tdetection model name should be one of "
        "{mobiledetv2-person-vehicle, "
        "mobiledetv2-person-pets, "
        "mobiledetv2-coco80, "
        "mobiledetv2-vehicle"
        "mobiledetv2-pedestrian}, default: mobiledetv2-coco80\n\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  if (argc == 6) {
    strcpy(args->model_name, argv[5]);
  } else {
    strcpy(args->model_name, "mobiledetv2-d0");
  }

  if (get_od_model_info(args->model_name, &args->od_model_id, &args->inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", args->model_name);
    return CVIAI_FAILURE;
  }

  int err;
  strcpy(args->model_path, argv[1]);
  if ((err = check_file(args->model_path)) != 0) {
    printf("check model fail: %s, errno: %d\n", args->model_path, err);
    return CVIAI_FAILURE;
  }

  strcpy(args->image_folder_path, argv[2]);
  if ((err = check_dir(args->image_folder_path)) != 0) {
    printf("check image folder fail: %s, errno: %d\n", args->image_folder_path, err);
    return CVIAI_FAILURE;
  }

  strcpy(args->eval_json_path, argv[3]);
  if ((err = check_file(args->eval_json_path)) != 0) {
    printf("check json fail: %s, errno: %d\n", args->eval_json_path, err);
    return CVIAI_FAILURE;
  }

  strcpy(args->regression_output_path, argv[4]);
  return CVIAI_SUCCESS;
}

int main(int argc, char *argv[]) {
  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;

  Argument args;
  ret = parse_args(argc, argv, &args);
  if (ret != CVIAI_SUCCESS) {
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

  ret = CVI_AI_OpenModel(ai_handle, args.od_model_id, args.model_path);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model yolov3 failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, args.od_model_id, false);
  CVI_AI_SetModelThreshold(ai_handle, args.od_model_id, 0.05);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVIAI_SUCCESS) {
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
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &frame, PIXEL_FORMAT_RGB_888_PLANAR) != CVIAI_SUCCESS) {
      printf("Read image failed.\n");
      break;
    }
    free(filename);
    cvai_object_t obj;
    args.inference(ai_handle, &frame, &obj);

    for (int j = 0; j < obj.size; j++) {
      obj.info[j].classes = obj.info[j].classes + 1;
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
