#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"
#include "od_utils.h"

typedef struct _Argument {
  char model_path[1024];
  ODInferenceFunc inference;
  CVI_TDL_SUPPORTED_MODEL_E od_model_id;
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
    return CVI_TDL_FAILURE;
  }

  if (argc == 6) {
    strcpy(args->model_name, argv[5]);
  } else {
    strcpy(args->model_name, "mobiledetv2-d0");
  }

  if (get_od_model_info(args->model_name, &args->od_model_id, &args->inference) ==
      CVI_TDL_FAILURE) {
    printf("unsupported model: %s\n", args->model_name);
    return CVI_TDL_FAILURE;
  }

  int err;
  strcpy(args->model_path, argv[1]);
  if ((err = check_file(args->model_path)) != 0) {
    printf("check model fail: %s, errno: %d\n", args->model_path, err);
    return CVI_TDL_FAILURE;
  }

  strcpy(args->image_folder_path, argv[2]);
  if ((err = check_dir(args->image_folder_path)) != 0) {
    printf("check image folder fail: %s, errno: %d\n", args->image_folder_path, err);
    return CVI_TDL_FAILURE;
  }

  strcpy(args->eval_json_path, argv[3]);
  if ((err = check_file(args->eval_json_path)) != 0) {
    printf("check json fail: %s, errno: %d\n", args->eval_json_path, err);
    return CVI_TDL_FAILURE;
  }

  strcpy(args->regression_output_path, argv[4]);
  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  CVI_TDL_PerfettoInit();
  CVI_S32 ret = CVI_TDL_SUCCESS;

  Argument args;
  ret = parse_args(argc, argv, &args);
  if (ret != CVI_TDL_SUCCESS) {
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
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, args.od_model_id, args.model_path);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model yolov3 failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(tdl_handle, args.od_model_id, false);
  CVI_TDL_SetModelThreshold(tdl_handle, args.od_model_id, 0.05);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_TDL_Eval_CocoInit(eval_handle, args.image_folder_path, args.eval_json_path, &image_num);
  CVI_TDL_Eval_CocoStartEval(eval_handle, args.regression_output_path);
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_TDL_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);

    printf("[%d/%d] Reading image %s\n", i + 1, image_num, filename);
    VIDEO_FRAME_INFO_S frame;
    if (CVI_TDL_ReadImage(img_handle, filename, &frame, PIXEL_FORMAT_RGB_888_PLANAR) !=
        CVI_TDL_SUCCESS) {
      printf("Read image failed.\n");
      break;
    }
    free(filename);
    cvtdl_object_t obj;
    args.inference(tdl_handle, &frame, &obj);

    for (int j = 0; j < obj.size; j++) {
      obj.info[j].classes = obj.info[j].classes + 1;
    }

    CVI_TDL_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_TDL_Free(&obj);
    CVI_TDL_ReleaseImage(img_handle, &frame);
  }
  CVI_TDL_Eval_CocoEndEval(eval_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_SYS_Exit();
}
