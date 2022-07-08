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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define OUTPUT_BUFFER_SIZE 10
#define MODE_DEFINITION 0

// #define VISUAL_FACE_LANDMARK
// #define USE_OUTPUT_DATA_API

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
  cvai_image_t image;
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

/* global variables */
static volatile bool bExit = false;
static volatile bool bRunImageWriter = true;
static volatile bool bRunVideoOutput = true;

int rear_idx = 0;
int front_idx = 0;
static IOData data_buffer[OUTPUT_BUFFER_SIZE];

static cvai_face_t g_face_meta_0;
static cvai_face_t g_face_meta_1;

static APP_MODE_e app_mode;

/* helper functions */
bool READ_CONFIG(const char *config_path, face_capture_config_t *app_config);

bool CHECK_OUTPUT_CONDITION(face_capture_t *face_cpt_info, uint32_t idx, APP_MODE_e mode);

/**
 * Restructure the face meta of the face capture to 2 output face struct.
 * 0: Low quality, 1: Otherwise (Ignore unstable trackers)
 */
void RESTRUCTURING_FACE_META(face_capture_t *face_cpt_info, cvai_face_t *face_meta_0,
                             cvai_face_t *face_meta_1);

int COUNT_ALIVE(face_capture_t *face_cpt_info);

#ifdef VISUAL_FACE_LANDMARK
void FREE_FACE_PTS(cvai_face_t *face_meta);
#endif

#ifdef USE_OUTPUT_DATA_API
uint32_t GENERATE_OUTPUT_DATA(IOData **output_data, face_capture_t *face_cpt_info);
void FREE_OUTPUT_DATA(IOData *output_data, uint32_t size);
#endif

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
    if (data_buffer[target_idx].image.pix_format != PIXEL_FORMAT_RGB_888) {
      printf("[WARNING] Image I/O unsupported format: %d\n",
             data_buffer[target_idx].image.pix_format);
    } else {
      if (data_buffer[target_idx].image.width == 0) {
        printf("[WARNING] Target image is empty.\n");
      } else {
        printf(" > (I/O) Write Face (Q: %.2f): %s ...\n", data_buffer[target_idx].quality,
               filename);
        stbi_write_png(filename, data_buffer[target_idx].image.width,
                       data_buffer[target_idx].image.height, STBI_rgb,
                       data_buffer[target_idx].image.pix[0],
                       data_buffer[target_idx].image.stride[0]);

        /* if there is no first capture face in INTELLIGENT mode, we need to create one */
        if (app_mode == intelligent && data_buffer[target_idx].counter == 0) {
          sprintf(filename, "images/face_%" PRIu64 "_1.png", data_buffer[target_idx].u_id);
          stbi_write_png(filename, data_buffer[target_idx].image.width,
                         data_buffer[target_idx].image.height, STBI_rgb,
                         data_buffer[target_idx].image.pix[0],
                         data_buffer[target_idx].image.stride[0]);
        }
      }
    }

    free(filename);
    CVI_AI_Free(&data_buffer[target_idx].image);
    {
      SMT_MutexAutoLock(IOMutex, lock);
      front_idx = target_idx;
    }
  }

  printf("[APP] free buffer data...\n");
  while (front_idx != rear_idx) {
    CVI_AI_Free(&data_buffer[(front_idx + 1) % OUTPUT_BUFFER_SIZE].image);
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
#ifdef VISUAL_FACE_LANDMARK
        face_meta_0.info[i].pts.size = g_face_meta_0.info[i].pts.size;
        face_meta_0.info[i].pts.x = (float *)malloc(sizeof(float) * face_meta_0.info[i].pts.size);
        face_meta_0.info[i].pts.y = (float *)malloc(sizeof(float) * face_meta_0.info[i].pts.size);
        memcpy(face_meta_0.info[i].pts.x, g_face_meta_0.info[i].pts.x,
               sizeof(float) * face_meta_0.info[i].pts.size);
        memcpy(face_meta_0.info[i].pts.y, g_face_meta_0.info[i].pts.y,
               sizeof(float) * face_meta_0.info[i].pts.size);
#endif
      }
      for (uint32_t i = 0; i < g_face_meta_1.size; i++) {
        face_meta_1.info[i].unique_id = g_face_meta_1.info[i].unique_id;
        face_meta_1.info[i].face_quality = g_face_meta_1.info[i].face_quality;
        memcpy(&face_meta_1.info[i].bbox, &g_face_meta_1.info[i].bbox, sizeof(cvai_bbox_t));
#ifdef VISUAL_FACE_LANDMARK
        face_meta_1.info[i].pts.size = g_face_meta_1.info[i].pts.size;
        face_meta_1.info[i].pts.x = (float *)malloc(sizeof(float) * face_meta_1.info[i].pts.size);
        face_meta_1.info[i].pts.y = (float *)malloc(sizeof(float) * face_meta_1.info[i].pts.size);
        memcpy(face_meta_1.info[i].pts.x, g_face_meta_1.info[i].pts.x,
               sizeof(float) * face_meta_1.info[i].pts.size);
        memcpy(face_meta_1.info[i].pts.y, g_face_meta_1.info[i].pts.y,
               sizeof(float) * face_meta_1.info[i].pts.size);
#endif
      }
    }

    size_t image_size = stVOFrame.stVFrame.u32Length[0] + stVOFrame.stVFrame.u32Length[1] +
                        stVOFrame.stVFrame.u32Length[2];
    stVOFrame.stVFrame.pu8VirAddr[0] =
        (uint8_t *)CVI_SYS_MmapCache(stVOFrame.stVFrame.u64PhyAddr[0], image_size);
    stVOFrame.stVFrame.pu8VirAddr[1] =
        stVOFrame.stVFrame.pu8VirAddr[0] + stVOFrame.stVFrame.u32Length[0];
    stVOFrame.stVFrame.pu8VirAddr[2] =
        stVOFrame.stVFrame.pu8VirAddr[1] + stVOFrame.stVFrame.u32Length[1];

    CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_0, &stVOFrame, false, brush_0);
    CVI_AI_Service_FaceDrawRect(service_handle, &face_meta_1, &stVOFrame, false, brush_1);
#ifdef VISUAL_FACE_LANDMARK
    CVI_AI_Service_FaceDraw5Landmark(&face_meta_0, &stVOFrame);
    CVI_AI_Service_FaceDraw5Landmark(&face_meta_1, &stVOFrame);
#endif

#if 1
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
#endif

    CVI_SYS_Munmap((void *)stVOFrame.stVFrame.pu8VirAddr[0], image_size);
    stVOFrame.stVFrame.pu8VirAddr[0] = NULL;
    stVOFrame.stVFrame.pu8VirAddr[1] = NULL;
    stVOFrame.stVFrame.pu8VirAddr[2] = NULL;

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

#ifdef VISUAL_FACE_LANDMARK
    FREE_FACE_PTS(&face_meta_0);
    FREE_FACE_PTS(&face_meta_1);
#endif
    CVI_AI_Free(&face_meta_0);
    CVI_AI_Free(&face_meta_1);
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc != 13) {
    printf(
        "Usage: %s fd model type, 0: normal, 1: mask face\n"
        "          fr model type, 0: recognition, 1: attribute\n"
        "          <face_detection_model_path>\n"
        "          <face_recognition_model_path> (NULL: disable FR)\n"
        "          <face_quality_model_path> (NULL: disable FQ)\n"
        "          <config_path>\n"
        "          mode, 0: fast, 1: interval, 2: leave, 3: intelligent\n"
        "          tracking buffer size\n"
        "          FD threshold\n"
        "          write image (0/1)\n"
        "          video output, 0: disable, 1: output to panel, 2: output through rtsp"
        "          video input format, 0: rgb, 1: nv21, 2: yuv420, 3: rgb(planar)\n",
        argv[0]);
    return CVIAI_FAILURE;
  }
  CVI_S32 ret = CVIAI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);
  int fd_model_type = atoi(argv[1]);
  int fr_model_type = atoi(argv[2]);
  const char *fd_model_path = argv[3];
  const char *fr_model_path = argv[4];
  const char *fq_model_path = argv[5];
  const char *config_path = argv[6];
  const char *mode_id = argv[7];
  int buffer_size = atoi(argv[8]);
  float det_threshold = atof(argv[9]);
  bool write_image = atoi(argv[10]) == 1;
  int voType = atoi(argv[11]);
  int vi_format = atoi(argv[12]);

  CVI_AI_SUPPORTED_MODEL_E fd_model_id = (fd_model_type == 0)
                                             ? CVI_AI_SUPPORTED_MODEL_RETINAFACE
                                             : CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION;
  CVI_AI_SUPPORTED_MODEL_E fr_model_id = (fr_model_type == 0)
                                             ? CVI_AI_SUPPORTED_MODEL_FACERECOGNITION
                                             : CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE;

  if (buffer_size <= 0) {
    printf("buffer size must be larger than 0.\n");
    return CVI_FAILURE;
  }

  VideoSystemContext vs_ctx = {0};
  SIZE_S aiInputSize = {.u32Width = 1280, .u32Height = 720};

  PIXEL_FORMAT_E aiInputFormat;
  if (vi_format == 0) {
    aiInputFormat = PIXEL_FORMAT_RGB_888;
  } else if (vi_format == 1) {
    aiInputFormat = PIXEL_FORMAT_NV21;
  } else if (vi_format == 2) {
    aiInputFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  } else if (vi_format == 3) {
    aiInputFormat = PIXEL_FORMAT_RGB_888_PLANAR;
  } else {
    printf("vi format[%d] unknown.\n", vi_format);
    return CVI_FAILURE;
  }
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
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, fd_model_id, fr_model_id, fd_model_path,
                                           (!strcmp(fr_model_path, "NULL")) ? NULL : fr_model_path,
                                           (!strcmp(fq_model_path, "NULL")) ? NULL : fq_model_path);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    goto CLEANUP_SYSTEM;
  }
  CVI_AI_SetVpssTimeout(ai_handle, 1000);

  CVI_AI_SetModelThreshold(ai_handle, fd_model_id, det_threshold);

  app_mode = atoi(mode_id);
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

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(config_path, "NULL")) {
    printf("Use Default Config...\n");
  } else {
    printf("Read Specific Config: %s\n", config_path);
    if (!READ_CONFIG(config_path, &app_cfg)) {
      printf("[ERROR] Read Config Failed.\n");
      goto CLEANUP_SYSTEM;
    }
  }
  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);

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

    ret = CVI_VPSS_GetChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI, &stfdFrame,
                               2000);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VPSS_GetChnFrame chn0 failed with %#x\n", ret);
      break;
    }

    int alive_face_num = COUNT_ALIVE(app_handle->face_cpt_info);
    printf("ALIVE Faces: %d\n", alive_face_num);

    ret = CVI_AI_APP_FaceCapture_Run(app_handle, &stfdFrame);
    if (ret != CVIAI_SUCCESS) {
      printf("CVI_AI_APP_FaceCapture_Run failed with %#x\n", ret);
      break;
    }

    {
      SMT_MutexAutoLock(VOMutex, lock);
#ifdef VISUAL_FACE_LANDMARK
      FREE_FACE_PTS(&g_face_meta_0);
      FREE_FACE_PTS(&g_face_meta_1);
#endif
      CVI_AI_Free(&g_face_meta_0);
      CVI_AI_Free(&g_face_meta_1);
      RESTRUCTURING_FACE_META(app_handle->face_cpt_info, &g_face_meta_0, &g_face_meta_1);
    }

    /* Producer */
    if (write_image) {
      for (uint32_t i = 0; i < app_handle->face_cpt_info->size; i++) {
        if (!CHECK_OUTPUT_CONDITION(app_handle->face_cpt_info, i, app_mode)) {
          continue;
        }
        tracker_state_e state = app_handle->face_cpt_info->data[i].state;
        uint32_t counter = app_handle->face_cpt_info->data[i]._out_counter;
        uint64_t u_id = app_handle->face_cpt_info->data[i].info.unique_id;
        float face_quality = app_handle->face_cpt_info->data[i].info.face_quality;
        if (state == MISS) {
          printf("Produce Face-%" PRIu64 "_out\n", u_id);
        } else {
          printf("Produce Face-%" PRIu64 "_%u\n", u_id, counter);
        }
        /* Check output buffer space */
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
        /* Copy image data to buffer */
        data_buffer[target_idx].u_id = u_id;
        data_buffer[target_idx].quality = face_quality;
        data_buffer[target_idx].state = state;
        data_buffer[target_idx].counter = counter;
        /* NOTE: Make sure the image type is IVE_IMAGE_TYPE_U8C3_PACKAGE */
        CVI_AI_CopyImage(&app_handle->face_cpt_info->data[i].image, &data_buffer[target_idx].image);
        {
          SMT_MutexAutoLock(IOMutex, lock);
          rear_idx = target_idx;
        }
      }
    }

    /* Generate output image data */
#ifdef USE_OUTPUT_DATA_API
    IOData *sample_output_data = NULL;
    uint32_t output_num = GENERATE_OUTPUT_DATA(&sample_output_data, app_handle->face_cpt_info);
    printf("Output Data (Size = %u)\n", output_num);
    for (uint32_t i = 0; i < output_num; i++) {
      printf("face[%u] ID: %" PRIu64 ", Quality: %.4f, Size: (%hu,%hu)\n", i,
             sample_output_data[i].u_id, sample_output_data[i].quality,
             sample_output_data[i].image.height, sample_output_data[i].image.width);
    }
    FREE_OUTPUT_DATA(sample_output_data, output_num);
#endif

    ret = CVI_VPSS_ReleaseChnFrame(vs_ctx.vpssConfigs.vpssGrp, vs_ctx.vpssConfigs.vpssChnAI,
                                   &stfdFrame);
    if (ret != CVI_SUCCESS) {
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
    } else if (!strcmp(name, "Capture_Aligned_Face")) {
      app_config->capture_aligned_face = atoi(value) == 1;
    } else if (!strcmp(name, "Capture_Extended_Face")) {
      app_config->capture_extended_face = atoi(value) == 1;
    } else if (!strcmp(name, "Store_Face_Feature")) {
      app_config->store_feature = atoi(value) == 1;
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
#ifdef VISUAL_FACE_LANDMARK
    (*tmp_ptr)->pts.size = face_cpt_info->last_faces.info[i].pts.size;
    (*tmp_ptr)->pts.x = (float *)malloc(sizeof(float) * (*tmp_ptr)->pts.size);
    (*tmp_ptr)->pts.y = (float *)malloc(sizeof(float) * (*tmp_ptr)->pts.size);
    memcpy((*tmp_ptr)->pts.x, face_cpt_info->last_faces.info[i].pts.x,
           sizeof(float) * (*tmp_ptr)->pts.size);
    memcpy((*tmp_ptr)->pts.y, face_cpt_info->last_faces.info[i].pts.y,
           sizeof(float) * (*tmp_ptr)->pts.size);
#endif
  }
  return;
}

bool CHECK_OUTPUT_CONDITION(face_capture_t *face_cpt_info, uint32_t idx, APP_MODE_e mode) {
  if (!face_cpt_info->_output[idx]) return false;
  if (mode == leave && face_cpt_info->data[idx].state != MISS) return false;
  return true;
}

#ifdef VISUAL_FACE_LANDMARK
void FREE_FACE_PTS(cvai_face_t *face_meta) {
  if (face_meta->size == 0) {
    return;
  }
  for (uint32_t j = 0; j < face_meta->size; j++) {
    face_meta->info[j].pts.size = 0;
    free(face_meta->info[j].pts.x);
    free(face_meta->info[j].pts.y);
    face_meta->info[j].pts.x = NULL;
    face_meta->info[j].pts.y = NULL;
  }
}
#endif

#ifdef USE_OUTPUT_DATA_API
uint32_t GENERATE_OUTPUT_DATA(IOData **output_data, face_capture_t *face_cpt_info) {
  uint32_t output_num = 0;
  for (uint32_t i = 0; i < face_cpt_info->size; i++) {
    if (CHECK_OUTPUT_CONDITION(face_cpt_info, i, app_mode)) {
      output_num += 1;
    }
  }
  if (output_num == 0) {
    *output_data = NULL;
    return 0;
  }
  *output_data = (IOData *)malloc(sizeof(IOData) * output_num);
  memset(*output_data, 0, sizeof(IOData) * output_num);

  uint32_t tmp_idx = 0;
  for (uint32_t i = 0; i < face_cpt_info->size; i++) {
    if (!CHECK_OUTPUT_CONDITION(face_cpt_info, i, app_mode)) {
      continue;
    }
    tracker_state_e state = face_cpt_info->data[i].state;
    uint32_t counter = face_cpt_info->data[i]._out_counter;
    uint64_t u_id = face_cpt_info->data[i].info.unique_id;
    float face_quality = face_cpt_info->data[i].info.face_quality;
    /* Copy image data to buffer */
    (*output_data)[tmp_idx].u_id = u_id;
    (*output_data)[tmp_idx].quality = face_quality;
    (*output_data)[tmp_idx].state = state;
    (*output_data)[tmp_idx].counter = counter;
    /* NOTE: Make sure the image type is IVE_IMAGE_TYPE_U8C3_PACKAGE */
    CVI_AI_CopyImage(&face_cpt_info->data[i].image, &(*output_data)[tmp_idx].image);
    tmp_idx += 1;
  }
  return output_num;
}

void FREE_OUTPUT_DATA(IOData *output_data, uint32_t size) {
  if (size == 0) return;
  for (uint32_t i = 0; i < size; i++) {
    CVI_AI_Free(&output_data[i].image);
  }
  free(output_data);
}
#endif