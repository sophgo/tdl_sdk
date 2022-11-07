#include "app/cviai_app.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"
// #include "vi_vo_utils.h"

#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vi.h>

#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sstream>
#include <string>
#include "ive/ive.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sys_utils.hpp"
#define OUTPUT_BUFFER_SIZE 10
#define MODE_DEFINITION 0

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

// typedef struct {
//   CVI_S32 voType;
//   VideoSystemContext vs_ctx;
//   cviai_service_handle_t service_handle;
// } pVOArgs;

SMT_MUTEXAUTOLOCK_INIT(IOMutex);
SMT_MUTEXAUTOLOCK_INIT(VOMutex);

/* global variables */
static volatile bool bExit = false;
static volatile bool bRunImageWriter = true;
static volatile bool bRunVideoOutput = true;

int rear_idx = 0;
int front_idx = 0;
static IOData data_buffer[OUTPUT_BUFFER_SIZE];

static cvai_object_t g_obj_meta_0;
static cvai_object_t g_obj_meta_1;

static APP_MODE_e app_mode;
std::string g_out_dir;

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
    char filename[128];
    if ((app_mode == leave || app_mode == intelligent) && data_buffer[target_idx].state == MISS) {
      sprintf(filename, "%s/face_%" PRIu64 "_out.png", g_out_dir.c_str(),
              data_buffer[target_idx].u_id);
    } else {
      sprintf(filename, "%s/face_%" PRIu64 "_%u.png", g_out_dir.c_str(),
              data_buffer[target_idx].u_id, data_buffer[target_idx].counter);
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
          sprintf(filename, "%s/face_%" PRIu64 "_1.png", g_out_dir.c_str(),
                  data_buffer[target_idx].u_id);
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

std::string capobj_to_str(face_cpt_data_t *p_obj, float w, float h, int lb) {
  std::stringstream ss;
  // ss<<p_obj->_timestamp<<",4,";
  float ctx = (p_obj->info.bbox.x1 + p_obj->info.bbox.x2) / 2.0 / w;
  float cty = (p_obj->info.bbox.y1 + p_obj->info.bbox.y2) / 2.0 / h;
  float ww = (p_obj->info.bbox.x2 - p_obj->info.bbox.x1) / w;
  float hh = (p_obj->info.bbox.y2 - p_obj->info.bbox.y1) / h;
  ss << p_obj->_timestamp << "," << lb << "," << ctx << "," << cty << "," << ww << "," << hh << ","
     << p_obj->info.unique_id << "," << p_obj->info.bbox.score << "\n";
  return ss.str();
}
void export_tracking_info(face_capture_t *p_cap_info, const std::string &str_dst_dir, int frame_id,
                          float imgw, float imgh, int lb) {
  cvai_face_t *p_objinfo = &(p_cap_info->last_faces);
  if (p_objinfo->size == 0) return;
  char sz_dstf[128];
  sprintf(sz_dstf, "%s/%08d.txt", str_dst_dir.c_str(), frame_id);
  FILE *fp = fopen(sz_dstf, "w");

  for (uint32_t i = 0; i < p_objinfo->size; i++) {
    // if(p_objinfo->info[i].unique_id != 0){
    // sprintf( buf, "\nOD DB File Size = %" PRIu64 " bytes \t"
    char szinfo[128];
    float ctx = (p_objinfo->info[i].bbox.x1 + p_objinfo->info[i].bbox.x2) / 2 / imgw;
    float cty = (p_objinfo->info[i].bbox.y1 + p_objinfo->info[i].bbox.y2) / 2 / imgh;
    float ww = (p_objinfo->info[i].bbox.x2 - p_objinfo->info[i].bbox.x1) / imgw;
    float hh = (p_objinfo->info[i].bbox.y2 - p_objinfo->info[i].bbox.y1) / imgh;
    float score = p_objinfo->info[i].bbox.score;
    sprintf(szinfo, "%d %f %f %f %f %" PRIu64 " %f\n", lb, ctx, cty, ww, hh,
            p_objinfo->info[i].unique_id, score);
    fwrite(szinfo, 1, strlen(szinfo), fp);
    // }
  }
  fclose(fp);
}
void release_system(cviai_handle_t ai_handle, cviai_service_handle_t service_handle,
                    cviai_app_handle_t app_handle) {
  CVI_AI_APP_DestroyHandle(app_handle);
  if (service_handle != NULL) CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  // DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
}
int main(int argc, char *argv[]) {
  CVI_S32 ret = CVI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);

  std::string process_flag(argv[1]);
  // std::string str_model_file = join_path(str_model_root ,
  // std::string("yolox_RetinafaceMask_lm_432_768_int8_0705.cvimodel")); CVI_AI_SUPPORTED_MODEL_E
  // fd_model_id = CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION;

  CVI_AI_SUPPORTED_MODEL_E model;
  std::string modelf;
  if (process_flag == "retina") {
    model = CVI_AI_SUPPORTED_MODEL_RETINAFACE;
    modelf = std::string(
        "/mnt/data/admin1_data/AI_CV/cv182x/ai_models/output/cv182x/"
        "retinaface_mnet0.25_342_608.cvimodel");
  } else if (process_flag == "yolox") {
    model = CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION;
    modelf = std::string(
        "/mnt/data/admin1_data/AI_CV/cv182x/ai_models/retinaface_mask_classifier.cvimodel");
  } else {
    model = CVI_AI_SUPPORTED_MODEL_SCRFDFACE;
    modelf = std::string(
        "/mnt/data/admin1_data/AI_CV/cv182x/ai_models/scrfd_DW_conv_432_768_int8_2.cvimodel");
  }
  std::string str_model_file = modelf;
  CVI_AI_SUPPORTED_MODEL_E fd_model_id = model;

  const char *fd_model_path = str_model_file.c_str();
  // const char *reid_model_path = "NULL";//argv[3];//NULL
  const char *config_path = "NULL";  // argv[4];//NULL
  // const char *mode_id = intelligent;//argv[5];//leave=2,intelligent=3

  CVI_AI_SUPPORTED_MODEL_E fr_model_id = CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE;

  int buffer_size = 10;       // atoi(argv[6]);//10
  float det_threshold = 0.5;  // atof(argv[7]);//0.5
  bool write_image = true;    // 1
  std::string str_image_root(argv[2]);
  std::string str_dst_root = std::string(argv[3]);
  if (!create_directory(std::string(argv[3]))) {
    std::cout << "create directory:" << str_dst_root << " failed\n";
  }
  std::string str_dst_video =
      join_path(str_dst_root, get_directory_name(str_image_root) + std::string("_") + process_flag);
  if (!create_directory(str_dst_video)) {
    std::cout << "create directory:" << str_dst_video << " failed\n";
    // return CVI_FAILURE;
  }
  g_out_dir = str_dst_video;
  if (buffer_size <= 0) {
    printf("buffer size must be larger than 0.\n");
    return CVI_FAILURE;
  }

  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  cviai_app_handle_t app_handle = NULL;
  ret = CVI_AI_CreateHandle2(&ai_handle, 1, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  ret |= CVI_AI_Service_EnableTPUDraw(service_handle, true);
  ret |= CVI_AI_APP_CreateHandle(&app_handle, ai_handle);
  ret |= CVI_AI_APP_FaceCapture_Init(app_handle, (uint32_t)buffer_size);
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, fd_model_id, fr_model_id, fd_model_path,
                                           NULL, NULL);

  if (ret != CVI_SUCCESS) {
    release_system(ai_handle, service_handle, app_handle);
    return CVI_FAILURE;
  }

  CVI_AI_SetModelThreshold(ai_handle, fd_model_id, det_threshold);
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 3);
  app_mode = intelligent;  // APP_MODE_e(atoi(mode_id));
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
      release_system(ai_handle, service_handle, app_handle);
      return CVI_FAILURE;
  }

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(config_path, "NULL")) {
    printf("Use Default Config...\n");
  } else {
    release_system(ai_handle, service_handle, app_handle);
    return CVI_FAILURE;
  }
  app_cfg.thr_quality = 0.1;
  app_cfg.thr_quality_high = 0.95;
  app_cfg.thr_size_min = 20;
  app_cfg.miss_time_limit = 10;
  app_cfg.store_RGB888 = true;

  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);

  memset(&g_obj_meta_0, 0, sizeof(cvai_object_t));
  memset(&g_obj_meta_1, 0, sizeof(cvai_object_t));

  pthread_t io_thread;
  pthread_create(&io_thread, NULL, pImageWrite, NULL);
  const int face_label = 11;

  std::string cap_result = str_dst_video + std::string("/cap_result.log");
  FILE *fp = fopen(cap_result.c_str(), "w");
  // std::stringstream ss;
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  if (ive_handle == NULL) {
    printf("CreateHandle failed with %#x!\n", ret);
    release_system(ai_handle, service_handle, app_handle);
    return CVI_FAILURE;
  }
  int num_append = 0;
  for (int img_idx = 0; img_idx < 1000; img_idx++) {
    std::cout << "processing:" << img_idx << "/530\n";
    char szimg[256];
    sprintf(szimg, "%s/%08d.jpg", str_image_root.c_str(), img_idx);
    bool empty_img = false;
    IVE_IMAGE_S image = CVI_IVE_ReadImage(ive_handle, szimg, IVE_IMAGE_TYPE_U8C3_PLANAR);
    if (image.u16Width == 0) {
      std::cout << "read image failed:" << std::string(szimg) << std::endl;
      if (img_idx > 350) {
        empty_img = true;
        num_append++;
        image = CVI_IVE_ReadImage(ive_handle, "/mnt/data/admin1_data/black_1080p.jpg",
                                  IVE_IMAGE_TYPE_U8C3_PLANAR);
        if (image.u16Width == 0) {
          std::cout << "read black emptry failed:" << std::string(szimg) << std::endl;
          continue;
        }
      } else
        continue;
    }
    if (num_append > 30) {
      break;
    }
    VIDEO_FRAME_INFO_S fdFrame;
    ret = CVI_IVE_Image2VideoFrameInfo(&image, &fdFrame, false);
    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << std::string(szimg)
                << std::endl;

      continue;
    }

    int alive_person_num = COUNT_ALIVE(app_handle->face_cpt_info);
    printf("ALIVE persons: %d\n", alive_person_num);
    ret = CVI_AI_APP_FaceCapture_Run(app_handle, &fdFrame);
    if (ret != CVI_SUCCESS) {
      printf("CVI_AI_APP_PersonCapture_Run failed with %#x\n", ret);
      break;
    }

    // {
    //   SMT_MutexAutoLock(VOMutex, lock);
    //   CVI_AI_Free(&g_obj_meta_0);
    //   CVI_AI_Free(&g_obj_meta_1);
    //   RESTRUCTURING_OBJ_META(app_handle->face_cpt_info, &g_obj_meta_0, &g_obj_meta_1);
    // }
    /* Producer */
    if (write_image) {
      std::cout << "to export trackinginfo\n";
      if (!empty_img)
        export_tracking_info(app_handle->face_cpt_info, str_dst_video, img_idx,
                             fdFrame.stVFrame.u32Width, fdFrame.stVFrame.u32Height, face_label);
      std::cout << "to capture trackinginfo\n";
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
          continue;
        }
        std::string str_res =
            capobj_to_str(&app_handle->face_cpt_info->data[i], fdFrame.stVFrame.u32Width,
                          fdFrame.stVFrame.u32Height, face_label);
        // ss<<str_res;
        fwrite(str_res.c_str(), 1, str_res.length(), fp);
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

    CVI_VPSS_ReleaseChnFrame(0, 0, &fdFrame);
    CVI_SYS_FreeI(ive_handle, &image);
  }
  fclose(fp);
  CVI_IVE_DestroyHandle(ive_handle);
  bRunImageWriter = false;
  bRunVideoOutput = false;
  pthread_join(io_thread, NULL);
  // pthread_join(vo_thread, NULL);

  // CLEANUP_SYSTEM:
  CVI_AI_APP_DestroyHandle(app_handle);
  CVI_AI_Service_DestroyHandle(service_handle);
  CVI_AI_DestroyHandle(ai_handle);
  // DestroyVideoSystem(&vs_ctx);
  CVI_SYS_Exit();
  CVI_VB_Exit();
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
