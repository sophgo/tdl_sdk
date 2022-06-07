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
#include <cstdlib>
#include <vector>

#define OUTPUT_BUFFER_SIZE 10
#define MODE_DEFINITION 0

#define VISUAL_FRAME_NUMBER
#define VISUAL_INACTIVATE_TRACKER
#define VISUAL_UNSTABLE_TRACKER

typedef enum { fast = 0, interval, leave, intelligent } APP_MODE_e;

/* global variables */
static volatile bool bExit = false;

/* helper functions */
bool READ_CONFIG(const char *config_path, face_capture_config_t *app_config);

/**
 * Restructure the face meta of the face capture to 2 output face struct.
 * 0: Low quality, 1: Otherwise (Ignore unstable trackers)
 */
void RESTRUCTURING_FACE_META(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                             cvai_face_t *face_meta_1);

int COUNT_ALIVE(face_capture_t *face_cpt_info);

typedef struct {
  int R;
  int G;
  int B;
} COLOR_RGB_t;

COLOR_RGB_t GET_RANDOM_COLOR(uint64_t seed, int min = 64);
void GENERATE_VISUAL_COLOR(cvai_face_t *faces, cvai_tracker_t *trackers, COLOR_RGB_t *colors);

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    printf(
        "Usage: %s <face_detection_model_path>\n"
        "          <face_recognition_model_path>\n"
        "          <face_quality_model_path>\n"
        "          <config_path>\n"
        "          mode, 0: fast, 1: interval, 2: leave, 3: intelligent\n"
        "          use FQNet (0/1)\n"
        "          tracking buffer size\n"
        "          FD threshold\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  int buffer_size = atoi(argv[7]);
  if (buffer_size <= 0) {
    printf("buffer size must be larger than 0.\n");
    return CVI_FAILURE;
  }

  APP_MODE_e app_mode = static_cast<APP_MODE_e>(atoi(argv[5]));
  float fd_threshold = atof(argv[8]);
  CVI_S32 voType = atoi(argv[9]);

  CVI_S32 s32Ret = CVI_SUCCESS;
  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  PIXEL_FORMAT_E aiInputFormat = PIXEL_FORMAT_RGB_888;
  // PIXEL_FORMAT_E aiInputFormat = PIXEL_FORMAT_NV21;
  if (InitVideoSystem(&vs_ctx, &aiInputSize, aiInputFormat, voType) != CVI_SUCCESS) {
    printf("failed to init video system\n");
    return CVI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  cviai_app_handle_t app_handle = NULL;

  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  ret |= CVI_AI_APP_CreateHandle(&app_handle, ai_handle);
  ret |= CVI_AI_APP_FaceCapture_Init(app_handle, (uint32_t)buffer_size);
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, argv[1], argv[2], argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    goto CLEANUP_SYSTEM;
  }
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_threshold);

  {
    switch (app_mode) {
#if MODE_DEFINITION == 0
      case fast: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, FAST);
      } break;
      case interval: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, CYCLE);
      } break;
      case leave: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, AUTO);
      } break;
      case intelligent: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, AUTO);
      } break;
#elif MODE_DEFINITION == 1
      case high_quality: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, AUTO);
      } break;
      case quick: {
        CVI_AI_APP_FaceCapture_SetMode(app_handle, FAST);
      } break;
#else
#error "Unexpected value of MODE_DEFINITION."
#endif
      default:
        printf("Unknown mode %d\n", app_mode);
        goto CLEANUP_SYSTEM;
    }
  }

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(argv[4], "NULL")) {
    printf("Use Default Config...\n");
  } else {
    printf("Read Specific Config: %s\n", argv[4]);
    if (!READ_CONFIG(argv[4], &app_cfg)) {
      printf("[ERROR] Read Config Failed.\n");
      goto CLEANUP_SYSTEM;
    }
  }
  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);

  app_handle->face_cpt_info->use_fqnet = atoi(argv[6]) == 1;

  VIDEO_FRAME_INFO_S stVIFrame;
  VIDEO_FRAME_INFO_S stVOFrame;

  uint32_t frame_counter;
  frame_counter = 0;
  while (bExit == false) {
    frame_counter += 1;
    printf("\nFrame counter[%u]\n", frame_counter);

    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                  &stVIFrame, 2000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    int alive_face_num = COUNT_ALIVE(app_handle->face_cpt_info);
    printf("ALIVE Faces: %d\n", alive_face_num);

    CVI_AI_APP_FaceCapture_Run(app_handle, &stVIFrame);
    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stVIFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }

    s32Ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnVideoOutput,
                                  &stVOFrame, 1000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    uint32_t image_size = stVOFrame.stVFrame.u32Length[0] + stVOFrame.stVFrame.u32Length[1] +
                          stVOFrame.stVFrame.u32Length[2];
    stVOFrame.stVFrame.pu8VirAddr[0] =
        (uint8_t *)CVI_SYS_MmapCache(stVOFrame.stVFrame.u64PhyAddr[0], image_size);
    stVOFrame.stVFrame.pu8VirAddr[1] =
        stVOFrame.stVFrame.pu8VirAddr[0] + stVOFrame.stVFrame.u32Length[0];
    stVOFrame.stVFrame.pu8VirAddr[2] =
        stVOFrame.stVFrame.pu8VirAddr[1] + stVOFrame.stVFrame.u32Length[1];

#ifdef VISUAL_FRAME_NUMBER
    char frame_number[8];
    sprintf(frame_number, "[%u]", frame_counter);
    CVI_AI_Service_ObjectWriteText(frame_number, 32, 32, &stVOFrame, 1, 1, 1);
#endif

    cvai_face_t face;
    face.size = 1;
    face.height = app_handle->face_cpt_info->last_faces.height;
    face.width = app_handle->face_cpt_info->last_faces.width;
    face.rescale_type = app_handle->face_cpt_info->last_faces.rescale_type;

#ifdef VISUAL_INACTIVATE_TRACKER
    cvai_face_info_t tmp_face_info;
    face.info = &tmp_face_info;
    COLOR_RGB_t INACTIVATE_TRACKER_COLOR;
    INACTIVATE_TRACKER_COLOR = {128, 128, 128};
    cvai_tracker_t inact_trackers;
    memset(&inact_trackers, 0, sizeof(cvai_tracker_t));
    CVI_AI_DeepSORT_GetTracker_Inactive(ai_handle, &inact_trackers);
    for (uint32_t i = 0; i < inact_trackers.size; i++) {
      cvai_service_brush_t brush;
      brush.color = {(float)INACTIVATE_TRACKER_COLOR.R, (float)INACTIVATE_TRACKER_COLOR.G,
                     (float)INACTIVATE_TRACKER_COLOR.B};
      brush.size = 2;
      tmp_face_info.bbox = inact_trackers.info[i].bbox;
      CVI_AI_Service_FaceDrawRect(service_handle, &face, &stVOFrame, false, brush);
      char id_num[64];
      sprintf(id_num, "%" PRIu64 "", inact_trackers.info[i].id);
      CVI_AI_Service_ObjectWriteText(id_num, face.info[0].bbox.x1, face.info[0].bbox.y1, &stVOFrame,
                                     (float)INACTIVATE_TRACKER_COLOR.R / 255.,
                                     (float)INACTIVATE_TRACKER_COLOR.G / 255.,
                                     (float)INACTIVATE_TRACKER_COLOR.B / 255.);
    }
    CVI_AI_Free(&inact_trackers);
#endif

    uint32_t face_size = app_handle->face_cpt_info->last_faces.size;
    COLOR_RGB_t *colors = (COLOR_RGB_t *)malloc(face_size * sizeof(COLOR_RGB_t));
    GENERATE_VISUAL_COLOR(&app_handle->face_cpt_info->last_faces,
                          &app_handle->face_cpt_info->last_trackers, colors);

    for (uint32_t i = 0; i < face_size; i++) {
#ifndef VISUAL_UNSTABLE_TRACKER
      if (app_handle->face_cpt_info->last_trackers.info[i].state != CVI_TRACKER_STABLE) continue;
#endif
      face.info = &app_handle->face_cpt_info->last_faces.info[i];
      cvai_service_brush_t brush;
      brush.color = {(float)colors[i].R, (float)colors[i].G, (float)colors[i].B};
      brush.size = 2;
      CVI_AI_Service_FaceDrawRect(service_handle, &face, &stVOFrame, false, brush);
      char id_num[64];
      sprintf(id_num, "%" PRIu64 "", face.info[0].unique_id);
      CVI_AI_Service_ObjectWriteText(id_num, face.info[0].bbox.x1, face.info[0].bbox.y1, &stVOFrame,
                                     (float)colors[i].R / 255., (float)colors[i].G / 255.,
                                     (float)colors[i].B / 255.);
    }

    CVI_SYS_Munmap((void *)stVOFrame.stVFrame.pu8VirAddr[0], image_size);
    stVOFrame.stVFrame.pu8VirAddr[0] = NULL;
    stVOFrame.stVFrame.pu8VirAddr[1] = NULL;
    stVOFrame.stVFrame.pu8VirAddr[2] = NULL;

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

CLEANUP_SYSTEM:
  CVI_AI_APP_DestroyHandle(app_handle);
  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}

#define CHAR_SIZE 64
bool READ_CONFIG(const char *config_path, face_capture_config_t *app_config) {
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
    } else if (!strcmp(name, "Threshold_Size_Min")) {
      app_config->thr_size_min = atoi(value);
    } else if (!strcmp(name, "Threshold_Size_Max")) {
      app_config->thr_size_max = atoi(value);
    } else if (!strcmp(name, "Quality_Assessment_Method")) {
      app_config->qa_method = atoi(value);
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
    } else if (!strcmp(name, "AUTO_Mode_Time_Limit")) {
      app_config->auto_m_time_limit = (uint32_t)atoi(value);
    } else if (!strcmp(name, "AUTO_Mode_Fast_Cap")) {
      app_config->auto_m_fast_cap = atoi(value) == 1;
    } else if (!strcmp(name, "Do_Face_Recognition")) {
      app_config->do_FR = atoi(value) == 1;
    } else if (!strcmp(name, "Capture_Aligned_Face")) {
      app_config->capture_aligned_face = atoi(value) == 1;
    } else if (!strcmp(name, "Store_RGB888")) {
      app_config->store_RGB888 = atoi(value) == 1;
    } else {
      printf("Unknow Arg: %s\n", name);
      return false;
    }
  }
  fclose(fp);

  return true;
}

int COUNT_ALIVE(face_capture_t *face_cpt_info) {
  int counter = 0;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      counter += 1;
    }
  }
  return counter;
}

void RESTRUCTURING_FACE_META(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                             cvai_face_t *face_meta_1) {
  face_meta_0->size = 0;
  face_meta_1->size = 0;
  for (uint32_t i = 0; i < face_cpt_info->last_faces.size; i++) {
    if (face_cpt_info->last_trackers.info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    if (face_cpt_info->last_faces.info[i].face_quality >= face_cpt_info->cfg.thr_quality) {
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
    bool qualified =
        face_cpt_info->last_faces.info[i].face_quality >= face_cpt_info->cfg.thr_quality;
    cvai_face_info_t **tmp_ptr = (qualified) ? &info_ptr_1 : &info_ptr_0;
    (*tmp_ptr)->unique_id = face_cpt_info->last_faces.info[i].unique_id;
    (*tmp_ptr)->face_quality = face_cpt_info->last_faces.info[i].face_quality;
    memcpy(&(*tmp_ptr)->bbox, &face_cpt_info->last_faces.info[i].bbox, sizeof(cvai_bbox_t));
    *tmp_ptr += 1;
  }
  return;
}

COLOR_RGB_t GET_RANDOM_COLOR(uint64_t seed, int min) {
  float scale = (256. - (float)min) / 256.;
  srand(static_cast<uint32_t>(seed));
  COLOR_RGB_t color;
  color.R = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  color.G = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  color.B = (int)((floor(((float)rand() / (RAND_MAX)) * 256.)) * scale) + min;
  return color;
}

void GENERATE_VISUAL_COLOR(cvai_face_t *faces, cvai_tracker_t *trackers, COLOR_RGB_t *colors) {
  uint32_t face_size = faces->size;
  for (uint32_t i = 0; i < face_size; i++) {
    if (trackers->info[i].state == CVI_TRACKER_NEW) {
      colors[i] = {255, 255, 255};
      continue;
    }
    COLOR_RGB_t tmp_color = GET_RANDOM_COLOR(faces->info[i].unique_id);
    if (trackers->info[i].state == CVI_TRACKER_UNSTABLE) {
      tmp_color.R = tmp_color.R / 2;
      tmp_color.G = tmp_color.G / 2;
      tmp_color.B = tmp_color.B / 2;
    }
    colors[i] = tmp_color;
  }
}