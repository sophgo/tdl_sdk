#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"
#include "inttypes.h"
#include "od_utils.h"

int main(int argc, char *argv[]) {
  CVI_TDL_PerfettoInit();
  if (argc != 6) {
    printf(
        "Usage: %s <detection_model_name>\n"
        "          <detection_model_path>\n"
        "          <reid_model_path>\n"
        "          <sample_imagelist_path>\n"
        "          <MOT16_dataset_index(ex:MOT-03)>\n",
        argv[0]);
    return CVI_TDL_FAILURE;
  }
  CVI_S32 ret = CVI_TDL_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }
  cvitdl_handle_t tdl_handle = NULL;

  ODInferenceFunc inference;
  CVI_TDL_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(argv[1], &od_model_id, &inference) == CVI_TDL_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVI_TDL_FAILURE;
  }

  ret = CVI_TDL_CreateHandle2(&tdl_handle, 1, 0);

  ret = CVI_TDL_OpenModel(tdl_handle, od_model_id, argv[2]);
  ret |= CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, argv[3]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("model open failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(tdl_handle, od_model_id, false);
  CVI_TDL_SetSkipVpssPreprocess(tdl_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, false);

  // Init DeepSORT
  CVI_TDL_DeepSORT_Init(tdl_handle, false);
#if 1
  cvtdl_deepsort_config_t ds_conf;
  CVI_TDL_DeepSORT_GetDefaultConfig(&ds_conf);
  CVI_TDL_DeepSORT_SetConfig(tdl_handle, &ds_conf, -1, false);
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

    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_TDL_ReadImage(image_path, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_object_t obj_meta;
    cvtdl_tracker_t tracker_meta;
    memset(&obj_meta, 0, sizeof(cvtdl_object_t));
    memset(&tracker_meta, 0, sizeof(cvtdl_tracker_t));

    CVI_TDL_SelectDetectClass(tdl_handle, od_model_id, 1, CVI_TDL_DET_TYPE_PERSON);

    //*******************************************
    // Tracking function calls.
    // Step 1. Object detect inference.
    inference(tdl_handle, &frame, &obj_meta);
    // Step 2. Object feature generator.
    CVI_TDL_OSNet(tdl_handle, &frame, &obj_meta);
    // Step 3. Tracker.
    CVI_TDL_DeepSORT_Obj(tdl_handle, &obj_meta, &tracker_meta, true);
    // Tracking function calls ends here.
    //*******************************************
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      fprintf(outFile, "%d,%" PRIu64 ",%f,%f,%f,%f,%d,%d,%d,%d\n", counter,
              obj_meta.info[i].unique_id, obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
              obj_meta.info[i].bbox.x2 - obj_meta.info[i].bbox.x1,
              obj_meta.info[i].bbox.y2 - obj_meta.info[i].bbox.y1, 1, -1, -1, -1);
    }

    CVI_TDL_Free(&obj_meta);
    CVI_TDL_Free(&tracker_meta);
    CVI_TDL_ReleaseImage(&frame);
  }
  printf("\nDone\n");

  fclose(outFile);
  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_SYS_Exit();
}