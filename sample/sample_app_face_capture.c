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
#include "stb_image.h"
#include "stb_image_write.h"

#define OUTPUT_BUFFER_SIZE 10

typedef enum { fast = 0, interval, leave, intelligent } APP_MODE_e;

#define SMT_MUTEXAUTOLOCK_INIT(mutex) pthread_mutex_t AUTOLOCK_##mutex = PTHREAD_MUTEX_INITIALIZER;

#define SMT_MutexAutoLock(mutex, lock)                                            \
  __attribute__((cleanup(AutoUnLock))) pthread_mutex_t *lock = &AUTOLOCK_##mutex; \
  pthread_mutex_lock(lock);

__attribute__((always_inline)) inline void AutoUnLock(void *mutex) {
  pthread_mutex_unlock(*(pthread_mutex_t **)mutex);
}

typedef struct {
  uint64_t u_id;
  float quality;
  CVI_U16 height;
  CVI_U16 width;
  CVI_U16 stride;
  uint8_t *arr;
  tracker_state_e state;
  uint32_t counter;
} IOData;

typedef struct {
  CVI_S32 voType;
  VideoSystemContext vs_ctx;
  cviai_service_handle_t service_handle;
} pVOArgs;

SMT_MUTEXAUTOLOCK_INIT(IOMutex);
SMT_MUTEXAUTOLOCK_INIT(VOMutex);

static volatile bool bExit = false;
static volatile bool bRunImageWriter = true;
static volatile bool bRunVideoOutput = true;

int rear_idx = 0;
int front_idx = 0;
static IOData data_buffer[OUTPUT_BUFFER_SIZE];

static cvai_face_t g_face_meta_0;
static cvai_face_t g_face_meta_1;

static APP_MODE_e app_mode;

int get_alive_num(face_capture_t *face_cpt_info);
bool read_config(const char *config_path, face_capture_config_t *app_config);
/* 0: low quality, 1: otherwise (Note: ignore unstable trackers) */
void gen_face_meta_01(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                      cvai_face_t *face_meta_1);

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}

/* Consumer */
static void *pImageWrite(void *args) {
  printf("[APP] Image Write Up\n");
  while (bRunImageWriter) {
    /* only consumer write front_idx */
    bool empty;
    {
      SMT_MutexAutoLock(IOMutex, lock);
#if 0
      int remain = (rear_idx >= front_idx) ? rear_idx - front_idx
                                           : (rear_idx + 1) + (OUTPUT_BUFFER_SIZE - 1) - front_idx;
      printf("[DEBUG] BUFFER REMAIN: %d (front: %d, rear: %d)\n", remain, front_idx, rear_idx);
#endif
      empty = front_idx == rear_idx;
    }
    if (empty) {
      printf("I/O Buffer is empty.\n");
      usleep(100 * 1000);
      continue;
    }
    int target_idx = (front_idx + 1) % OUTPUT_BUFFER_SIZE;
    char *filename = calloc(64, sizeof(char));
    if ((app_mode == leave || app_mode == intelligent) && data_buffer[target_idx].state == MISS) {
      sprintf(filename, "images/face_%" PRIu64 "_out.png", data_buffer[target_idx].u_id);
    } else {
      sprintf(filename, "images/face_%" PRIu64 "_%u.png", data_buffer[target_idx].u_id,
              data_buffer[target_idx].counter);
    }
    printf(" > (I/O) Write Face (Q: %.2f): %s ...\n", data_buffer[target_idx].quality, filename);
    stbi_write_png(filename, data_buffer[target_idx].width, data_buffer[target_idx].height,
                   STBI_rgb, data_buffer[target_idx].arr, data_buffer[target_idx].stride);

    /* if there is no first capture face in INTELLIGENT mode, we need to create one (unnecessary) */
    if (app_mode == intelligent && data_buffer[target_idx].counter == 0) {
      sprintf(filename, "images/face_%" PRIu64 "_1.png", data_buffer[target_idx].u_id);
      stbi_write_png(filename, data_buffer[target_idx].width, data_buffer[target_idx].height,
                     STBI_rgb, data_buffer[target_idx].arr, data_buffer[target_idx].stride);
    }

    free(filename);
    free(data_buffer[target_idx].arr);
    {
      SMT_MutexAutoLock(IOMutex, lock);
      front_idx = target_idx;
    }
  }

  printf("[APP] free buffer data...\n");
  while (front_idx != rear_idx) {
    free(data_buffer[(front_idx + 1) % OUTPUT_BUFFER_SIZE].arr);
    {
      SMT_MutexAutoLock(IOMutex, lock);
      front_idx = (front_idx + 1) % OUTPUT_BUFFER_SIZE;
    }
  }

  return NULL;
}

static void *pVideoOutput(void *args) {
  printf("[APP] Video Output Up\n");
  pVOArgs *vo_args = (pVOArgs *)args;
  if (!vo_args->voType) {
    return NULL;
  }
  cviai_service_handle_t service_handle = vo_args->service_handle;
  CVI_S32 s32Ret = CVI_SUCCESS;

  cvai_service_brush_t brush_0 = {.size = 4, .color.r = 0, .color.g = 64, .color.b = 255};
  cvai_service_brush_t brush_1 = {.size = 8, .color.r = 0, .color.g = 255, .color.b = 0};

  cvai_face_t face_meta_0;
  cvai_face_t face_meta_1;
  memset(&face_meta_0, 0, sizeof(cvai_face_t));
  memset(&face_meta_1, 0, sizeof(cvai_face_t));

  VIDEO_FRAME_INFO_S stVOFrame;
  while (bRunVideoOutput) {
    s32Ret = CVI_VPSS_GetChnFrame(vo_args->vs_ctx.vpssConfigs.vpssGrp,
                                  vo_args->vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame, 1000);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", s32Ret);
      break;
    }

    {
      SMT_MutexAutoLock(VOMutex, lock);
      memcpy(&face_meta_0, &g_face_meta_0, sizeof(cvai_face_t));
      memcpy(&face_meta_1, &g_face_meta_1, sizeof(cvai_face_t));
      face_meta_0.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * g_face_meta_0.size);
      face_meta_1.info = (cvai_face_info_t *)malloc(sizeof(cvai_face_info_t) * g_face_meta_1.size);
      memset(face_meta_0.info, 0, sizeof(cvai_face_info_t) * face_meta_0.size);
      memset(face_meta_1.info, 0, sizeof(cvai_face_info_t) * face_meta_1.size);
      for (uint32_t i = 0; i < g_face_meta_0.size; i++) {
        face_meta_0.info[i].unique_id = g_face_meta_0.info[i].unique_id;
        face_meta_0.info[i].face_quality = g_face_meta_0.info[i].face_quality;
        memcpy(&face_meta_0.info[i].bbox, &g_face_meta_0.info[i].bbox, sizeof(cvai_bbox_t));
      }
      for (uint32_t i = 0; i < g_face_meta_1.size; i++) {
        face_meta_1.info[i].unique_id = g_face_meta_1.info[i].unique_id;
        face_meta_1.info[i].face_quality = g_face_meta_1.info[i].face_quality;
        memcpy(&face_meta_1.info[i].bbox, &g_face_meta_1.info[i].bbox, sizeof(cvai_bbox_t));
      }
    }

    CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_0, &stVOFrame, false, brush_0);
    CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_1, &stVOFrame, false, brush_1);

    for (uint32_t j = 0; j < face_meta_0.size; j++) {
      char *id_num = calloc(64, sizeof(char));
      // sprintf(id_num, "%" PRIu64 "", face_meta_0.info[j].unique_id);
      sprintf(id_num, "[%" PRIu64 "] %.4f", face_meta_0.info[j].unique_id,
              face_meta_0.info[j].face_quality);
      CVI_AI_Service_ObjectWriteText(id_num, face_meta_0.info[j].bbox.x1,
                                     face_meta_0.info[j].bbox.y1, &stVOFrame, 1, 1, 1);
      free(id_num);
    }
    for (uint32_t j = 0; j < face_meta_1.size; j++) {
      char *id_num = calloc(64, sizeof(char));
      // sprintf(id_num, "%" PRIu64 "", face_meta_1.info[j].unique_id);
      sprintf(id_num, "[%" PRIu64 "] %.4f", face_meta_1.info[j].unique_id,
              face_meta_1.info[j].face_quality);
      CVI_AI_Service_ObjectWriteText(id_num, face_meta_1.info[j].bbox.x1,
                                     face_meta_1.info[j].bbox.y1, &stVOFrame, 1, 1, 1);
      free(id_num);
    }

    s32Ret = SendOutputFrame(&stVOFrame, &vo_args->vs_ctx.outputContext);
    if (s32Ret != CVI_SUCCESS) {
      printf("Send Output Frame NG\n");
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(vo_args->vs_ctx.vpssConfigs.vpssGrp,
                                      vo_args->vs_ctx.vpssConfigs.vpssChnVideoOutput, &stVOFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }
    CVI_AI_Free(&face_meta_0);
    CVI_AI_Free(&face_meta_1);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 10) {
    printf(
        "Usage: %s <face_detection_model_path>\n"
        "          <face_quality_model_path>\n"
        "          <config_path>\n"
        "          mode, 0: fast, 1: interval, 2: leave, 3: intelligent\n"
        "          use FQNet (0/1)\n"
        "          tracking buffer size\n"
        "          FD threshold\n"
        "          write image (0/1)\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  int buffer_size = atoi(argv[6]);
  if (buffer_size <= 0) {
    printf("buffer size must be larger than 0.\n");
    return CVI_FAILURE;
  }

  CVI_S32 voType = atoi(argv[9]);

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
  ret |= CVI_AI_APP_FaceCapture_Init(app_handle, (uint32_t)buffer_size);
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, argv[1], argv[2]);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    goto CLEANUP_SYSTEM;
  }
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  float fd_threshold = atof(argv[7]);
  CVI_AI_SetModelThreshold(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_threshold);

  app_mode = atoi(argv[4]);
  switch (app_mode) {
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
    default:
      printf("Unknown mode %d\n", app_mode);
      goto CLEANUP_SYSTEM;
  }

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(argv[3], "NULL")) {
    printf("Use Default Config...\n");
  } else {
    printf("Read Specific Config: %s\n", argv[3]);
    if (!read_config(argv[3], &app_cfg)) {
      printf("[ERROR] Read Config Failed.\n");
      goto CLEANUP_SYSTEM;
    }
  }
  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);

  app_handle->face_cpt_info->use_fqnet = atoi(argv[5]) == 1;
  bool write_image = atoi(argv[8]) == 1;

  VIDEO_FRAME_INFO_S stfdFrame;

  memset(&g_face_meta_0, 0, sizeof(cvai_face_t));
  memset(&g_face_meta_1, 0, sizeof(cvai_face_t));

  pthread_t io_thread, vo_thread;
  pthread_create(&io_thread, NULL, pImageWrite, NULL);
  pVOArgs vo_args = {0};
  vo_args.voType = voType;
  vo_args.service_handle = service_handle;
  vo_args.vs_ctx = vs_ctx;
  pthread_create(&vo_thread, NULL, pVideoOutput, (void *)&vo_args);

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

    {
      SMT_MutexAutoLock(VOMutex, lock);
      CVI_AI_Free(&g_face_meta_0);
      CVI_AI_Free(&g_face_meta_1);
      gen_face_meta_01(app_handle->face_cpt_info, &g_face_meta_0, &g_face_meta_1);
    }

    /* Producer */
    if (write_image) {
      for (uint32_t i = 0; i < app_handle->face_cpt_info->size; i++) {
        if (!app_handle->face_cpt_info->_output[i]) {
          continue;
        }
        tracker_state_e state = app_handle->face_cpt_info->data[i].state;
        if (app_mode == leave && state != MISS) {
          continue;
        }
        uint64_t u_id = app_handle->face_cpt_info->data[i].info.unique_id;
        float face_quality = app_handle->face_cpt_info->data[i].info.face_quality;
        uint32_t counter = app_handle->face_cpt_info->data[i]._out_counter;
        if (state == MISS) {
          printf("Produce Face-%" PRIu64 "_out\n", u_id);
        } else {
          printf("Produce Face-%" PRIu64 "_%u\n", u_id, counter);
        }
        /* copy image data to buffer */
        bool full;
        int target_idx;
        {
          SMT_MutexAutoLock(IOMutex, lock);
          target_idx = (rear_idx + 1) % OUTPUT_BUFFER_SIZE;
          full = target_idx == front_idx;
        }
        if (full) {
          printf("[WARNING] Buffer is full! Drop out!");
          continue;
        }
        /* assert image type is IVE_IMAGE_TYPE_U8C3_PACKAGE */
        CVI_U16 height = app_handle->face_cpt_info->data[i].height;
        CVI_U16 width = app_handle->face_cpt_info->data[i].width;
        CVI_U16 stride = app_handle->face_cpt_info->data[i].stride;
        data_buffer[target_idx].height = height;
        data_buffer[target_idx].width = width;
        data_buffer[target_idx].stride = stride;
        data_buffer[target_idx].arr = (uint8_t *)malloc(stride * height);
        memcpy(data_buffer[target_idx].arr, app_handle->face_cpt_info->data[i].face_pix,
               stride * height * sizeof(CVI_U8));
        data_buffer[target_idx].u_id = u_id;
        data_buffer[target_idx].quality = face_quality;
        data_buffer[target_idx].state = state;
        data_buffer[target_idx].counter = counter;
        {
          SMT_MutexAutoLock(IOMutex, lock);
          rear_idx = target_idx;
        }
      }
    }

    s32Ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                      &stfdFrame);
    if (s32Ret != CVI_SUCCESS) {
      printf("CVI_VPSS_ReleaseChnFrame chn0 NG\n");
      break;
    }
  }
  bRunImageWriter = false;
  bRunVideoOutput = false;
  pthread_join(io_thread, NULL);
  pthread_join(vo_thread, NULL);

CLEANUP_SYSTEM:
  CVI_AI_APP_DestroyHandle(app_handle);
  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  CVI_IVE_DestroyHandle(ive_handle);
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
    } else if (!strcmp(name, "Threshold_Size")) {
      app_config->thr_size = atoi(value);
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