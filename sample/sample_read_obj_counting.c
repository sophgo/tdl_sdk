/**
 * This is a sample code for object counting. Tracking targets contain person, car.
 */

#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#include <inttypes.h>

#define WRITE_RESULT_TO_FILE 0
#define TARGET_NUM 2

typedef int (*InferenceFunc)(cviai_handle_t, VIDEO_FRAME_INFO_S *, cvai_object_t *);
typedef struct _ModelConfig {
  CVI_AI_SUPPORTED_MODEL_E model_id;
  int input_size;
  InferenceFunc inference;
} ModelConfig;

CVI_S32 createModelConfig(const char *model_name, ModelConfig *config) {
  CVI_S32 ret = CVIAI_SUCCESS;

  if (strcmp(model_name, "mobiledetv2-coco80") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80;
    config->inference = CVI_AI_MobileDetV2_COCO80;
  } else if (strcmp(model_name, "yolov3") == 0) {
    config->model_id = CVI_AI_SUPPORTED_MODEL_YOLOV3;
    config->inference = CVI_AI_Yolov3;
  } else {
    ret = CVIAI_FAILURE;
  }
  return ret;
}

void setSampleMOTConfig(cvai_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.accreditation_threshold = 10;
  ds_conf->ktracker_conf.P_std_beta[2] = 0.01;
  ds_conf->ktracker_conf.P_std_beta[6] = 1e-5;
  ds_conf->kfilter_conf.Q_std_beta[2] = 0.01;
  ds_conf->kfilter_conf.Q_std_beta[6] = 1e-5;
  ds_conf->kfilter_conf.R_std_beta[2] = 0.1;
}

typedef struct {
  int classes_id[TARGET_NUM];  // {CVI_AI_DET_TYPE_PERSON, CVI_AI_DET_TYPE_CAR, ...}
  uint64_t classes_count[TARGET_NUM];
  uint64_t classes_maxID[TARGET_NUM];
} obj_counter_t;

// Not completed
/* stable tracking counter */
void update_obj_counter_stable(obj_counter_t *obj_counter, cvai_object_t *obj_meta,
                               cvai_tracker_t *tracker_meta) {
  uint64_t new_maxID[TARGET_NUM];
  uint64_t newID_num[TARGET_NUM];
  memset(newID_num, 0, sizeof(uint64_t) * TARGET_NUM);
  // Add the number of IDs which greater than original maxID.
  // Finally update new maxID for each counter.
  for (int j = 0; j < TARGET_NUM; j++) {
    new_maxID[j] = obj_counter->classes_maxID[j];
  }

  for (uint32_t i = 0; i < obj_meta->size; i++) {
    // Skip the bbox whoes tracker state is not stable
    if (tracker_meta->info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    // Find the index of object counter for this class
    int class_index = -1;
    for (int j = 0; j < TARGET_NUM; j++) {
      if (obj_meta->info[i].classes == obj_counter->classes_id[j]) {
        class_index = j;
        break;
      }
    }
    if (obj_meta->info[i].unique_id > obj_counter->classes_maxID[class_index]) {
      newID_num[class_index] += 1;
      if (obj_meta->info[i].unique_id > new_maxID[class_index]) {
        new_maxID[class_index] = obj_meta->info[i].unique_id;
      }
    }
  }

  for (int j = 0; j < TARGET_NUM; j++) {
    obj_counter->classes_count[j] += newID_num[j];
    obj_counter->classes_maxID[j] = new_maxID[j];
  }
}

/* simple tracking counter */
void update_obj_counter_simple(obj_counter_t *obj_counter, cvai_object_t *obj_meta) {
  for (uint32_t i = 0; i < obj_meta->size; i++) {
    // Find the index of object counter for this class
    int class_index = -1;
    for (int j = 0; j < TARGET_NUM; j++) {
      if (obj_meta->info[i].classes == obj_counter->classes_id[j]) {
        class_index = j;
        break;
      }
    }
    if (obj_meta->info[i].unique_id > obj_counter->classes_count[class_index]) {
      obj_counter->classes_count[class_index] = obj_meta->info[i].unique_id;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <detection_model_name>\n"
        "          <detection_model_path>\n"
        "          <reid_model_path>\n"
        "          <use_stable_counter (0/1)>\n"
        "          <sample_imagelist_path>\n"
        "          <inference_count>\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;

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
  if (createModelConfig(argv[1], &model_config) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);

  ret = CVI_AI_SetModelPath(ai_handle, model_config.model_id, argv[2]);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[3]);
  if (ret != CVIAI_SUCCESS) {
    printf("model open failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(ai_handle, model_config.model_id, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);
  CVI_AI_SelectDetectClass(ai_handle, model_config.model_id, 2, CVI_AI_DET_TYPE_PERSON,
                           CVI_AI_DET_TYPE_CAR);

  obj_counter_t obj_counter;
  memset(&obj_counter, 0, sizeof(obj_counter_t));
  obj_counter.classes_id[0] = CVI_AI_DET_TYPE_PERSON;
  obj_counter.classes_id[0] = CVI_AI_DET_TYPE_CAR;

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, true);
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  setSampleMOTConfig(&ds_conf);
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);

#if WRITE_RESULT_TO_FILE
  FILE *outFile;
  outFile = fopen("sample_ObjCounting_result.txt", "w");
  if (outFile == NULL) {
    printf("There is a problem opening the output file.\n");
    exit(EXIT_FAILURE);
  }
#endif

  int use_stable_counter = atoi(argv[4]);

  char *imagelist_path = argv[5];
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

#if WRITE_RESULT_TO_FILE
  fprintf(outFile, "%u\n", imageNum);
#endif

  int inference_count = atoi(argv[6]);

  for (int counter = 0; counter < imageNum; counter++) {
    if (counter == inference_count) {
      break;
    }
    if ((read = getline(&line, &len, inFile)) == -1) {
      printf("get line error\n");
      exit(EXIT_FAILURE);
    }
    *strchrnul(line, '\n') = '\0';
    char *image_path = line;
    printf("[%i] image path = %s\n", counter, image_path);

    IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

    // Read image using IVE.
    IVE_IMAGE_S ive_frame = CVI_IVE_ReadImage(ive_handle, image_path, IVE_IMAGE_TYPE_U8C3_PACKAGE);
    if (ive_frame.u16Width == 0) {
      printf("Read image failed with %x!\n", ret);
      return ret;
    }
    // Convert to VIDEO_FRAME_INFO_S. IVE_IMAGE_S must be kept to release when not used.
    VIDEO_FRAME_INFO_S frame;
    ret = CVI_IVE_Image2VideoFrameInfo(&ive_frame, &frame, false);
    if (ret != CVI_SUCCESS) {
      printf("Convert to video frame failed with %#x!\n", ret);
      return ret;
    }

    cvai_object_t obj_meta;
    cvai_tracker_t tracker_meta;
    memset(&obj_meta, 0, sizeof(cvai_object_t));
    memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

    //*******************************************
    // Step 1: Object detect inference.
    model_config.inference(ai_handle, &frame, &obj_meta);
    // Step 2: Extract ReID feature for all person bbox.
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      if (obj_meta.info[i].classes == CVI_AI_DET_TYPE_PERSON) {
        CVI_AI_OSNetOne(ai_handle, &frame, &obj_meta, (int)i);
      }
    }
    // Step 3: Multi-Object Tracking inference.
    CVI_AI_DeepSORT_Obj(ai_handle, &obj_meta, &tracker_meta, true);
    //*******************************************

    if (use_stable_counter) {
      update_obj_counter_stable(&obj_counter, &obj_meta, &tracker_meta);
    } else {
      update_obj_counter_simple(&obj_counter, &obj_meta);
    }

    for (int i = 0; i < TARGET_NUM; i++) {
      printf("[%d] %" PRIu64 "\n", obj_counter.classes_id[i], obj_counter.classes_count[i]);
    }

#if WRITE_RESULT_TO_FILE
    fprintf(outFile, "%u\n", tracker_meta.size);
    for (uint32_t i = 0; i < tracker_meta.size; i++) {
      fprintf(outFile, "%d,%" PRIu64 ",%d,%d,%d,%d,%d,%d,%d,%d,%d\n", obj_meta.info[i].classes,
              obj_meta.info[i].unique_id, (int)obj_meta.info[i].bbox.x1,
              (int)obj_meta.info[i].bbox.y1, (int)obj_meta.info[i].bbox.x2,
              (int)obj_meta.info[i].bbox.y2, tracker_meta.info[i].state,
              (int)tracker_meta.info[i].bbox.x1, (int)tracker_meta.info[i].bbox.y1,
              (int)tracker_meta.info[i].bbox.x2, (int)tracker_meta.info[i].bbox.y2);
    }
    fprintf(outFile, "%d\n", TARGET_NUM);
    for (int i = 0; i < TARGET_NUM; i++) {
      fprintf(outFile, "%d,%" PRIu64 "\n", obj_counter.classes_id[i], obj_counter.classes_count[i]);
    }
    fprintf(outFile, "%u\n", 0);
    // char debug_info[8192];
    // CVI_AI_DeepSORT_DebugInfo_1(ai_handle, debug_info);
    // fprintf(outFile, debug_info);
#endif

    CVI_AI_Free(&obj_meta);
    CVI_AI_Free(&tracker_meta);
    CVI_SYS_FreeI(ive_handle, &ive_frame);
    CVI_IVE_DestroyHandle(ive_handle);
  }

#if WRITE_RESULT_TO_FILE
  fclose(outFile);
#endif
  CVI_AI_DestroyHandle(ai_handle);
  CVI_SYS_Exit();
}
