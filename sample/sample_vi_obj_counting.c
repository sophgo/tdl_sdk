/**
 * This is a sample code for object counting. Tracking targets contain person, car.
 */

#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
#include "sample_utils.h"
#include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WRITE_RESULT_TO_FILE 1
#define TARGET_NUM 2

static volatile bool bExit = false;

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

void setSampleMOTConfig(cvai_deepsort_config_t *ds_conf) {
  ds_conf->ktracker_conf.accreditation_threshold = 10;
  ds_conf->ktracker_conf.P_beta[2] = 0.01;
  ds_conf->ktracker_conf.P_beta[6] = 1e-5;
  ds_conf->kfilter_conf.Q_beta[2] = 0.01;
  ds_conf->kfilter_conf.Q_beta[6] = 1e-5;
  ds_conf->kfilter_conf.R_beta[2] = 0.1;
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
  if (argc != 6) {
    printf(
        "Usage: %s <detection_model_name>\n"
        "          <detection_model_path>\n"
        "          <reid_model_path>\n"
        "          <use_stable_counter (0/1)>\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[5]);
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1920, .u32Height = 1080};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, VI_PIXEL_FORMAT, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  CVI_S32 ret = CVIAI_SUCCESS;
  ODInferenceFunc inference;
  CVI_AI_SUPPORTED_MODEL_E od_model_id;
  if (get_od_model_info(argv[1], &od_model_id, &inference) == CVIAI_FAILURE) {
    printf("unsupported model: %s\n", argv[1]);
    return CVIAI_FAILURE;
  }
  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  GOTO_IF_FAILED(CVI_AI_CreateHandle2(&ai_handle, 1, 0), s32Ret, create_ai_fail);
  GOTO_IF_FAILED(CVI_AI_Service_CreateHandle(&service_handle, ai_handle), s32Ret,
                 create_service_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, od_model_id, argv[2]), s32Ret, setup_ai_fail);
  GOTO_IF_FAILED(CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[3]), s32Ret,
                 setup_ai_fail);

  GOTO_IF_FAILED(CVI_AI_SelectDetectClass(ai_handle, od_model_id, 2, CVI_AI_DET_TYPE_PERSON,
                                          CVI_AI_DET_TYPE_CAR),
                 s32Ret, setup_ai_fail);

  // Init DeepSORT
  CVI_AI_DeepSORT_Init(ai_handle, true);
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  setSampleMOTConfig(&ds_conf);
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);

  int use_stable_counter = atoi(argv[4]);
  if (use_stable_counter) {
    printf("Use Stable Counter.\n");
  }

  obj_counter_t obj_counter;
  memset(&obj_counter, 0, sizeof(obj_counter_t));
  obj_counter.classes_id[0] = CVI_AI_DET_TYPE_PERSON;
  obj_counter.classes_id[1] = CVI_AI_DET_TYPE_CAR;

  cvai_object_t obj_meta;
  cvai_tracker_t tracker_meta;
  memset(&obj_meta, 0, sizeof(cvai_object_t));
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

  VIDEO_FRAME_INFO_S stFrame, stVOFrame;
  size_t counter = 0;
  while (bExit == false) {
    counter += 1;
    printf("\nGet Frame %zu\n", counter);
    ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI, &stFrame,
                               2000);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", ret);
      break;
    }

    //*******************************************
    // Step 1: Object detect inference.
    inference(ai_handle, &stFrame, &obj_meta);
    // Step 2: Extract ReID feature for all person bbox.
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      if (obj_meta.info[i].classes == CVI_AI_DET_TYPE_PERSON) {
        CVI_AI_OSNetOne(ai_handle, &stFrame, &obj_meta, (int)i);
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

    ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                   &stFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnVideoOutput,
                                 &stVOFrame, 1000);
      if (ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", ret);
        break;
      }
      CVI_AI_Service_ObjectDrawRect(service_handle, &obj_meta, &stVOFrame, false,
                                    CVI_AI_Service_GetDefaultBrush());
      for (uint32_t i = 0; i < tracker_meta.size; i++) {
        char *obj_ID = calloc(64, sizeof(char));
        sprintf(obj_ID, "%" PRIu64 "", obj_meta.info[i].unique_id);
        CVI_AI_Service_ObjectWriteText(obj_ID, obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
                                       &stVOFrame, -1, -1, -1);
        free(obj_ID);
      }
      for (int i = 0; i < TARGET_NUM; i++) {
        char *counter_info = calloc(64, sizeof(char));
        sprintf(counter_info, "[%d] %" PRIu64 "", obj_counter.classes_id[i],
                obj_counter.classes_count[i]);
        CVI_AI_Service_ObjectWriteText(counter_info, 0, (i + 1) * 20, &stVOFrame, -1, -1, -1);
        free(counter_info);
      }
      ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                     vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }

    CVI_AI_Free(&obj_meta);
    CVI_AI_Free(&tracker_meta);
  }

setup_ai_fail:
  CVI_AI_Service_DestroyHandle(service_handle);
create_service_fail:
  CVI_AI_DestroyHandle(ai_handle);
create_ai_fail:
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
  return s32Ret;
}
