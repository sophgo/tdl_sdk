#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "inttypes.h"

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *,
                             cvai_obj_det_type_e);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVI_SUCCESS;

  if (strcmp(model_name, "mobiledetv2-d0") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0;
    config->inference = CVI_AI_MobileDetV2_D0;
  } else if (strcmp(model_name, "mobiledetv2-d1") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D1;
    config->inference = CVI_AI_MobileDetV2_D1;
  } else if (strcmp(model_name, "mobiledetv2-d2") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D2;
    config->inference = CVI_AI_MobileDetV2_D2;
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
  } else {
    ret = CVI_FAILURE;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  CVI_AI_PerfettoInit();
  if (argc != 6) {
    printf(
        "Usage: %s <detection_model_name>\n"
        "          <detection_model_path>\n"
        "          <reid_model_path>\n"
        "          <sample_imagelist_path>\n"
        "          <MOT16_dataset_index(ex:MOT-03)>\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cviai_handle_t ai_handle = NULL;

  ModelConfig model_config;
  if (createModelConfig(argv[1], &model_config) == CVI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1);

  ret = CVI_AI_SetModelPath(ai_handle, model_config.model_id, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("model open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, model_config.model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle);
#if 1
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf);
#endif

  FILE *outFile;
  char result_file_name[16];
  snprintf(result_file_name, 16, "%s%s", argv[5], ".txt");
  // strcat(result_file_name, argv[5]);
  // strcat(result_file_name, ".txt");
  printf("Out File: %s\n", result_file_name);
  outFile = fopen(result_file_name, "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file.\n");
    exit(EXIT_FAILURE);
  }

  char *imagelist_path = argv[4];
  FILE *inFile;
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  inFile = fopen(imagelist_path, "r");
  if (inFile == NULL) {
    printf("There is a problem opening the rcfile: %s\n", imagelist_path);
    exit(EXIT_FAILURE);
  }
  if ((read = getline(&line, &len, inFile)) == -1) {
    printf("get line error\n");
    exit(EXIT_FAILURE);
  }
  *strchrnul(line, '\n') = '\0';
  int imageNum = atoi(line);

  for (int counter = 1; counter <= imageNum; counter++) {
    if ((read = getline(&line, &len, inFile)) == -1) {
      printf("get line error\n");
      exit(EXIT_FAILURE);
    }
    *strchrnul(line, '\n') = '\0';
    char *image_path = line;
    printf("[%d/%d]\n", counter, imageNum);

    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(image_path, &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_object_t obj_meta;
    cvai_tracker_t tracker_meta;
    memset(&obj_meta, 0, sizeof(cvai_object_t));
    memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

    //*******************************************
    // Tracking function calls.
    // Step 1. Object detect inference.
    model_config.inference(ai_handle, &frame, &obj_meta, CVI_DET_TYPE_PEOPLE);
    // Step 2. Object feature generator.
    CVI_AI_OSNet(ai_handle, &frame, &obj_meta);
    // Step 3. Tracker.
    CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, true);
    // Tracking function calls ends here.
    //*******************************************
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      fprintf(outFile, "%d,%" PRIu64 ",%f,%f,%f,%f,%d,%d,%d,%d\n", counter,
              obj_meta.info[i].unique_id, obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
              obj_meta.info[i].bbox.x2 - obj_meta.info[i].bbox.x1,
              obj_meta.info[i].bbox.y2 - obj_meta.info[i].bbox.y1, 1, -1, -1, -1);
    }

    CVI_AI_Free(&obj_meta);
    CVI_AI_Free(&tracker_meta);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  printf("\nDone\n");

  fclose(outFile);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}