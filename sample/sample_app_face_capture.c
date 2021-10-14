#include "app/cviai_app.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
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

#include "ive/ive.h"

static volatile bool bExit = false;

#define DEFAULT_BUFFER_SIZE 10

uint32_t face_counter = 0;

int getNumDigits(uint64_t num);
char *uint64ToString(uint64_t number);
char *floatToString(float number);

int get_alive_num(face_capture_t *face_cpt_info);

// 0: The faces not capturing last time, 1: otherwise (Note: ignore unstable trackers)
void gen_face_meta_01(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                      cvai_face_t *face_meta_1);
void write_miss_faces(face_capture_t *face_cpt_info, IVE_HANDLE ive_handle);
bool read_config(const char *config_path, face_capture_config_t *app_config);

enum APP_MODE { fast = 0, interval, leave, intelligent };

int main(int argc, char *argv[]) {
  if (argc != 7) {
    printf(
        "Usage: %s <face_detection_model_path>\n"
        "          <face_quality_model_path>\n"
        "          <config_path>\n"
        "          mode, 0: fast, 1: interval, 2: leave, 3: intelligent\n"
        "          write image (0/1)\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;

  CVI_S32 voType = atoi(argv[6]);

  CVI_S32 s32Ret = CVIAI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  if (InitVideoSystem(&vs_ctx, &aiInputSize, PIXEL_FORMAT_RGB_888, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVIAI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  cviai_app_handle_t app_handle = NULL;
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  ret |= CVI_AI_Service_EnableTPUDraw(service_handle, true);
  ret |= CVI_AI_APP_CreateHandle(&app_handle, ai_handle, ive_handle);
  ret |= CVI_AI_APP_FaceCapture_Init(app_handle, DEFAULT_BUFFER_SIZE);
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, argv[1], argv[2]);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  enum APP_MODE app_mode = atoi(argv[4]);
  bool write_image = atoi(argv[5]) == 1;
  bool output_miss = false;
  switch (app_mode) {
    case fast: {
      CVI_AI_APP_FaceCapture_SetMode(app_handle, FAST);
      if (write_image) {
        app_handle->face_cpt_info->write_face_stable = true;
      }
    } break;
    case interval: {
      CVI_AI_APP_FaceCapture_SetMode(app_handle, CYCLE);
      if (write_image) {
        app_handle->face_cpt_info->write_face_init = true;
        app_handle->face_cpt_info->write_face_stable = true;
      }
    } break;
    case leave: {
      CVI_AI_APP_FaceCapture_SetMode(app_handle, AUTO);
      if (write_image) {
        output_miss = true;
      }
    } break;
    case intelligent: {
      CVI_AI_APP_FaceCapture_SetMode(app_handle, CYCLE);
      if (write_image) {
        app_handle->face_cpt_info->write_face_init = true;
        output_miss = true;
      }
    } break;
    default:
      printf("Unknown license type %d\n", app_mode);
      return CVI_FAILURE;
  }

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(argv[3], "NULL")) {
    printf("Use Default Config...\n");
  } else {
    printf("Read Specific Config: %s\n", argv[3]);
    if (!read_config(argv[3], &app_cfg)) {
      printf("[ERROR] Read Config Failed.\n");
      CVI_AI_APP_DestroyHandle(app_handle);
      CVI_AI_Service_DestroyHandle(service_handle);
      CVI_AI_DestroyHandle(ai_handle);
      DestroyVideoSystem(&vs_ctx);
      CVI_SYS_Exit();
      CVI_VB_Exit();
      return CVI_FAILURE;
    }
  }
  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);
  printf("Write Face Init: %d\n", app_handle->face_cpt_info->write_face_init);
  printf("Write Face Stable: %d\n", app_handle->face_cpt_info->write_face_stable);

  VIDEO_FRAME_INFO_S stfdFrame, stVOFrame;
  cvai_service_brush_t brush_0;
  cvai_service_brush_t brush_1;
  brush_0.size = 4;
  brush_0.color.r = 0;
  brush_0.color.g = 64;
  brush_0.color.b = 255;
  brush_1.size = 8;
  brush_1.color.r = 0;
  brush_1.color.g = 255;
  brush_1.color.b = 0;
  size_t counter = 0;
  while (bExit == false) {
    counter += 1;
    printf("\nGet Frame %zu\n", counter);

    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stfdFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    int trk_num = get_alive_num(app_handle->face_cpt_info);
    printf("ALIVE face num = %d\n", trk_num);

    CVI_AI_APP_FaceCapture_Run(app_handle, &stfdFrame);

    cvai_face_t face_meta_0;
    memset(&face_meta_0, 0, sizeof(cvai_face_t));
    cvai_face_t face_meta_1;
    memset(&face_meta_1, 0, sizeof(cvai_face_t));
    gen_face_meta_01(app_handle->face_cpt_info, &face_meta_0, &face_meta_1);

    if (output_miss) {
      write_miss_faces(app_handle->face_cpt_info, ive_handle);
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    // Send frame to VO if opened.
    if (voType) {
      s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                    vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
        break;
      }
      CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_0, &stVOFrame, false, brush_0);
      CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_1, &stVOFrame, false, brush_1);
      for (uint32_t j = 0; j < face_meta_0.size; j++) {
        char *id_num = uint64ToString(face_meta_0.info[j].unique_id);
        CVI_AI_Service_ObjectWriteText(id_num, face_meta_0.info[j].bbox.x1,
                                       face_meta_0.info[j].bbox.y1, &stVOFrame, 1, 1, 1);
        free(id_num);
      }
      for (uint32_t j = 0; j < face_meta_1.size; j++) {
        char *id_num = uint64ToString(face_meta_1.info[j].unique_id);
        CVI_AI_Service_ObjectWriteText(id_num, face_meta_1.info[j].bbox.x1,
                                       face_meta_1.info[j].bbox.y1, &stVOFrame, 1, 1, 1);
        free(id_num);
      }

      s32Ret = SendOutputFrame(&stVOFrame, &vs_ctx.outputContext);
      if (s32Ret != CVI_SUCCESS) {
        printf("Send Output Frame NG\n");
      }

      s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp,
                                        vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
      if (s32Ret != CVI_SUCCESS) {
        printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
        break;
      }
    }
    CVI_AI_Free(&face_meta_0);
    CVI_AI_Free(&face_meta_1);
  }
  CVI_AI_APP_DestroyHandle(app_handle);
  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}

#define CHAR_SIZE 64
bool read_config(const char *config_path, face_capture_config_t *app_config) {
  char name[CHAR_SIZE];
  char value[CHAR_SIZE];
  FILE *fp = fopen(config_path, "r");
  if (fp == NULL) {
    return false;
  }
  while (!feof(fp)) {
    memset(name, 0, CHAR_SIZE);
    memset(value, 0, CHAR_SIZE);
    /*Read Data*/
    fscanf(fp, "%s %s\n", name, value);
    if (!strcmp(name, "Miss_Time_Limit")) {
      app_config->miss_time_limit = (uint32_t)atoi(value);
    } else if (!strcmp(name, "Threshold_Quality")) {
      app_config->thr_quality = atof(value);
    } else if (!strcmp(name, "Threshold_Quality_High")) {
      app_config->thr_quality_high = atof(value);
    } else if (!strcmp(name, "Threshold_Yaw")) {
      app_config->thr_yaw = atof(value);
    } else if (!strcmp(name, "Threshold_Pitch")) {
      app_config->thr_pitch = atof(value);
    } else if (!strcmp(name, "Threshold_Roll")) {
      app_config->thr_roll = atof(value);
    } else if (!strcmp(name, "FAST_Mode_Interval")) {
      app_config->fast_m_interval = (uint32_t)atoi(value);
    } else if (!strcmp(name, "FAST_Mode_Capture_Num")) {
      app_config->fast_m_capture_num = (uint32_t)atoi(value);
    } else if (!strcmp(name, "CYCLE_Mode_Interval")) {
      app_config->cycle_m_interval = (uint32_t)atoi(value);
    } else {
      printf("Unknow Arg: %s\n", name);
      return false;
    }
  }
  fclose(fp);

  return true;
}

int getNumDigits(uint64_t num) {
  int digits = 0;
  do {
    num /= 10;
    digits++;
  } while (num != 0);
  return digits;
}

char *uint64ToString(uint64_t number) {
  int n = getNumDigits(number);
  int i;
  char *numArray = calloc(n, sizeof(char));
  for (i = n - 1; i >= 0; --i, number /= 10) {
    numArray[i] = (number % 10) + '0';
  }
  return numArray;
}

char *floatToString(float number) {
  char *numArray = calloc(64, sizeof(char));
  // sprintf(numArray, "%g", number);
  sprintf(numArray, "%.4f", number);
  return numArray;
}

int get_alive_num(face_capture_t *face_cpt_info) {
  int counter = 0;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      counter += 1;
    }
  }
  return counter;
}

void gen_face_meta_01(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                      cvai_face_t *face_meta_1) {
  face_meta_0->size = 0;
  face_meta_1->size = 0;
  for (uint32_t i = 0; i < face_cpt_info->last_faces.size; i++) {
    if (face_cpt_info->last_trackers.info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    if (face_cpt_info->last_capture[i]) {
      face_meta_1->size += 1;
    } else {
      face_meta_0->size += 1;
    }
  }

  face_meta_0->info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * face_meta_0->size);
  memset(face_meta_0->info, 0, sizeof(cvai_face_info_t) * face_meta_0->size);
  face_meta_0->rescale_type = face_cpt_info->last_faces.rescale_type;
  face_meta_0->height = face_cpt_info->last_faces.height;
  face_meta_0->width = face_cpt_info->last_faces.width;

  face_meta_1->info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * face_meta_1->size);
  memset(face_meta_1->info, 0, sizeof(cvai_face_info_t) * face_meta_1->size);
  face_meta_1->rescale_type = face_cpt_info->last_faces.rescale_type;
  face_meta_1->height = face_cpt_info->last_faces.height;
  face_meta_1->width = face_cpt_info->last_faces.width;

  cvai_face_info_t *info_ptr_0 = face_meta_0->info;
  cvai_face_info_t *info_ptr_1 = face_meta_1->info;
  for (uint32_t i = 0; i < face_cpt_info->last_faces.size; i++) {
    if (face_cpt_info->last_trackers.info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    cvai_face_info_t **tmp_ptr = (face_cpt_info->last_capture[i]) ? &info_ptr_1 : &info_ptr_0;
    (*tmp_ptr)->unique_id = face_cpt_info->last_faces.info[i].unique_id;
    memcpy(&(*tmp_ptr)->bbox, &face_cpt_info->last_faces.info[i].bbox, sizeof(cvai_bbox_t));
    *tmp_ptr += 1;
  }
  return;
}

void write_miss_faces(face_capture_t *face_cpt_info, IVE_HANDLE ive_handle) {
  for (uint32_t i = 0; i < face_cpt_info->size; i++) {
    if (face_cpt_info->data[i].state != MISS) {
      continue;
    }
    printf("Write MISS Face[%u]\n", i);
    char *filename = calloc(32, sizeof(char));
    face_counter += 1;
    sprintf(filename, "images/face_%" PRIu64 "_out.png", face_cpt_info->data[i].info.unique_id);
    printf("Write Face to: %s\n", filename);
    CVI_IVE_WriteImage(ive_handle, filename, &face_cpt_info->data[i].face_image);
  }
  return;
}