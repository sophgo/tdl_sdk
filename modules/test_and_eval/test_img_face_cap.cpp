#include "app/cviai_app.h"
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "sample_comm.h"

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
#ifdef CV181X
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif
#include "evaluation/cviai_media.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "sys_utils.hpp"
#define OUTPUT_BUFFER_SIZE 10
#define MODE_DEFINITION 0
#define FACE_FEAT_SIZE 256
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
  char name[128];
  float match_score;
  uint64_t frame_id;
} IOData;

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

int COUNT_ALIVE(face_capture_t *face_cpt_info);

#ifdef VISUAL_FACE_LANDMARK
void FREE_FACE_PTS(cvai_face_t *face_meta);
#endif

static void SampleHandleSig(CVI_S32 signo) {
  signal(SIGINT, SIG_IGN);
  signal(SIGTERM, SIG_IGN);

  if (SIGINT == signo || SIGTERM == signo) {
    bExit = true;
  }
}
CVI_S32 SAMPLE_COMM_VPSS_Stop(VPSS_GRP VpssGrp, CVI_BOOL *pabChnEnable) {
  CVI_S32 j;
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_CHN VpssChn;

  for (j = 0; j < VPSS_MAX_PHY_CHN_NUM; j++) {
    if (pabChnEnable[j]) {
      VpssChn = j;
      s32Ret = CVI_VPSS_DisableChn(VpssGrp, VpssChn);
      if (s32Ret != CVI_SUCCESS) {
        printf("failed with %#x!\n", s32Ret);
        return CVI_FAILURE;
      }
    }
  }

  s32Ret = CVI_VPSS_StopGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  s32Ret = CVI_VPSS_DestroyGrp(VpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  return CVI_SUCCESS;
}
/* Consumer */
static void *pImageWrite(void *args) {
  printf("[APP] pImageWrite thread up\n");
  while (bRunImageWriter) {
    /* only consumer write front_idx */
    bool empty;
    {
      SMT_MutexAutoLock(IOMutex, lock);
      empty = front_idx == rear_idx;
    }
    if (empty) {
      printf("I/O Buffer is empty.\n");
      usleep(100 * 1000);
      continue;
    }
    int target_idx = (front_idx + 1) % OUTPUT_BUFFER_SIZE;
    char filename[256];
    if (data_buffer[target_idx].image.width == 0) {
      printf("[WARNING] Target image is empty.\n");
    } else {
      if (data_buffer[target_idx].image.pix_format == PIXEL_FORMAT_RGB_888) {
        sprintf(filename, "%s/frm_%d_face_%d_%u_score_%.3f_qua_%.3f_name_%s.png", g_out_dir.c_str(),
                int(data_buffer[target_idx].frame_id), int(data_buffer[target_idx].u_id),
                data_buffer[target_idx].counter, data_buffer[target_idx].match_score,
                data_buffer[target_idx].quality, data_buffer[target_idx].name);
        stbi_write_png(filename, data_buffer[target_idx].image.width,
                       data_buffer[target_idx].image.height, STBI_rgb,
                       data_buffer[target_idx].image.pix[0],
                       data_buffer[target_idx].image.stride[0]);
      } else {
        printf("to output image format:%d,not :%d\n", (int)data_buffer[target_idx].image.pix_format,
               (int)PIXEL_FORMAT_RGB_888);
        sprintf(filename, "%s/frm_%d_face_%d_%u_score_%.3f_qua_%.3f_name_%s.bin", g_out_dir.c_str(),
                int(data_buffer[target_idx].frame_id), int(data_buffer[target_idx].u_id),
                data_buffer[target_idx].counter, data_buffer[target_idx].match_score,
                data_buffer[target_idx].quality, data_buffer[target_idx].name);
        CVI_AI_DumpImage(filename, &data_buffer[target_idx].image);
      }
    }

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
  ss << p_obj->cap_timestamp << "," << lb << "," << ctx << "," << cty << "," << ww << "," << hh
     << "," << p_obj->info.unique_id << "," << p_obj->info.bbox.score << "\n";
  return ss.str();
}
void export_tracking_info(face_capture_t *p_cap_info, const std::string &str_dst_dir, int frame_id,
                          float imgw, float imgh, int lb) {
  cvai_face_t *p_objinfo = &(p_cap_info->last_faces);
  if (p_objinfo->size == 0) return;
  char sz_dstf[128];
  sprintf(sz_dstf, "%s/%08d.txt", str_dst_dir.c_str(), frame_id);
  FILE *fp = fopen(sz_dstf, "w");
  std::cout << "to parse,numobjs:" << p_objinfo->size << ",filename:" << sz_dstf << std::endl;
  for (uint32_t i = 0; i < p_objinfo->size; i++) {
    // if(p_objinfo->info[i].unique_id != 0){
    // sprintf( buf, "\nOD DB File Size = %" PRIu64 " bytes \t"
    char szinfo[128];
    float ctx = (p_objinfo->info[i].bbox.x1 + p_objinfo->info[i].bbox.x2) / 2 / imgw;
    float cty = (p_objinfo->info[i].bbox.y1 + p_objinfo->info[i].bbox.y2) / 2 / imgh;
    float ww = (p_objinfo->info[i].bbox.x2 - p_objinfo->info[i].bbox.x1) / imgw;
    float hh = (p_objinfo->info[i].bbox.y2 - p_objinfo->info[i].bbox.y1) / imgh;
    // float score = p_objinfo->info[i].bbox.score;
    sprintf(szinfo, "%d %f %f %f %f %d %.3f\n", lb, ctx, cty, ww, hh,
            int(p_objinfo->info[i].unique_id), p_objinfo->info[i].face_quality);

    fwrite(szinfo, 1, strlen(szinfo), fp);
    // }
  }
  std::cout << "write done\n";
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
int load_image_file(IVE_HANDLE ive_handle, const std::string &strf, IVE_IMAGE_S &image,
                    VIDEO_FRAME_INFO_S &fdFrame, IVE_IMAGE_TYPE_E img_format) {
  int ret = CVI_SUCCESS;
#ifdef CV181X
  PIXEL_FORMAT_E pix_format = PIXEL_FORMAT_RGB_888_PLANAR;
  if (img_format == IVE_IMAGE_TYPE_U8C3_PACKAGE) {
    pix_format = PIXEL_FORMAT_RGB_888;
  }
  ret = CVI_AI_ReadImage(strf.c_str(), &fdFrame, pix_format);
  return ret;
#endif

  image = CVI_IVE_ReadImage(ive_handle, strf.c_str(), img_format);
#ifdef CV181X
  int imgw = image.u32Width;
#else
  int imgw = image.u16Width;
#endif
  if (imgw == 0) {
    std::cout << "read image failed:" << strf << std::endl;
    return CVI_FAILURE;
  } else {
    std::cout << "readimg with:" << imgw << std::endl;
  }

#ifdef CV181X
  ret = CVI_IVE_Image2VideoFrameInfo(&image, &fdFrame);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image, &fdFrame, false);
#endif
  if (ret != CVI_SUCCESS) {
    std::cout << "Convert to video frame failed with:" << ret << ",file:" << strf << std::endl;
    CVI_SYS_FreeI(ive_handle, &image);
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}
std::string get_img_name(const std::string &strf) {
  size_t pos0 = strf.find_last_of('/');
  size_t pos1 = strf.find_last_of('.');
  std::string name = strf.substr(pos0 + 1, pos1 - pos0);
  return name;
}
int register_gallery_face(cviai_app_handle_t app_handle, IVE_HANDLE ive_handle,
                          const std::string &strf, cvai_service_feature_array_t *p_feat_gallery,
                          std::vector<std::string> &gallery_names) {
  IVE_IMAGE_S image;
  VIDEO_FRAME_INFO_S fdFrame;

  int ret = load_image_file(ive_handle, strf, image, fdFrame, IVE_IMAGE_TYPE_U8C3_PLANAR);
  if (ret != CVI_SUCCESS) {
    return NULL;
  }
  cvai_face_t faceinfo;
  memset(&faceinfo, 0, sizeof(faceinfo));
  ret = CVI_AI_APP_FaceCapture_FDFR(app_handle, &fdFrame, &faceinfo);
  if (ret != CVI_SUCCESS) {
    std::cout << "face extract failed\n";
  }
  std::cout << "extract face num:" << faceinfo.size << std::endl;
  if (faceinfo.size == 0 || faceinfo.info[0].feature.ptr == NULL) {
    printf("face num error,got:%d\n", (int)faceinfo.size);
    ret = CVI_FAILURE;
  } else {
    std::cout << "extract featsize:" << faceinfo.info[0].feature.size
              << ",addr:" << (void *)faceinfo.info[0].feature.ptr << std::endl;
  }
  if (ret == CVI_FAILURE) {
    CVI_VPSS_ReleaseChnFrame(0, 0, &fdFrame);
    CVI_SYS_FreeI(ive_handle, &image);
    CVI_AI_Free(&faceinfo);
    return ret;
  }

  int8_t *p_new_feat = NULL;
  size_t src_size = 0;
  if (p_feat_gallery->ptr == 0) {
    p_new_feat = (int8_t *)malloc(faceinfo.info[0].feature.size);
    p_feat_gallery->type = faceinfo.info[0].feature.type;
    p_feat_gallery->feature_length = faceinfo.info[0].feature.size;
    std::cout << "allocate memory,size:" << p_feat_gallery->feature_length << std::endl;
  } else {
    if (p_feat_gallery->feature_length != faceinfo.info[0].feature.size) {
      printf("error,featsize not equal,curface:%u,gallery:%u\n", faceinfo.info[0].feature.size,
             p_feat_gallery->feature_length);
      ret = CVI_FAILURE;
    } else {
      src_size = p_feat_gallery->feature_length * p_feat_gallery->data_num;
      p_new_feat = (int8_t *)malloc(src_size + FACE_FEAT_SIZE);
      memcpy(p_new_feat, p_feat_gallery->ptr, src_size);
    }
  }
  std::cout << "to copy mem\n";
  if (ret == CVI_SUCCESS) {
    if (p_feat_gallery->ptr != NULL) {
      memcpy(p_new_feat, p_feat_gallery->ptr + src_size, p_feat_gallery->feature_length);
      free(p_feat_gallery->ptr);
    }
    memcpy(p_new_feat + src_size, faceinfo.info[0].feature.ptr, faceinfo.info[0].feature.size);
    p_feat_gallery->data_num += 1;
    p_feat_gallery->ptr = p_new_feat;
    gallery_names.push_back(get_img_name(strf));
    std::cout << "register gallery:" << gallery_names[gallery_names.size() - 1] << std::endl;
  }
  std::cout << "copy done\n";
  CVI_VPSS_ReleaseChnFrame(0, 0, &fdFrame);
  CVI_SYS_FreeI(ive_handle, &image);
  CVI_AI_Free(&faceinfo);
  std::cout << "register done\n";
  return ret;
}
int do_face_match(cviai_service_handle_t service_handle,
                  const std::vector<std::string> &gallery_names, cvai_face_info_t *p_face) {
  if (gallery_names.size() == 0) {
    return CVI_FAILURE;
  }
  std::cout << "to matchtrack:" << p_face->unique_id << ",featsize:" << p_face->feature.size
            << std::endl;
  if (p_face->feature.size == 0) {
    return CVI_FAILURE;
  }
  uint32_t ind = 0;
  float score = 0;
  uint32_t size;

  int ret = CVI_AI_Service_FaceInfoMatching(service_handle, p_face, 1, 0.1, &ind, &score, &size);
  // printf("ind:%u,ret:%d,score:%f\n",ind,ret,score);
  // getchar();
  printf("matchname,trackid:%u,name:%s,score:%f\n", uint32_t(p_face->unique_id),
         gallery_names[ind].c_str(), score);
  p_face->recog_score = score;
  if (score > 0.5) {
    sprintf(p_face->name, gallery_names[ind].c_str(), gallery_names[ind].size());
  }
  return ret;
}

int main(int argc, char *argv[]) {
  CVI_S32 ret = CVI_SUCCESS;
  // Set signal catch
  signal(SIGINT, SampleHandleSig);
  signal(SIGTERM, SampleHandleSig);
  CVI_BOOL abChnEnable[VPSS_MAX_CHN_NUM] = {
      CVI_TRUE,
  };

  for (VPSS_GRP VpssGrp = 0; VpssGrp < VPSS_MAX_GRP_NUM; ++VpssGrp)
    SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);

  CVI_AI_SUPPORTED_MODEL_E model = CVI_AI_SUPPORTED_MODEL_SCRFDFACE;
  std::string modelf = std::string(
      "/mnt/data/admin1_data/AI_CV/cv182x/ai_models_output/cv181x/"
      "scrfd_500m_bnkps_432_768.cvimodel");

  std::string str_model_file = modelf;
  CVI_AI_SUPPORTED_MODEL_E fd_model_id = model;

  const char *fd_model_path = str_model_file.c_str();

  const char *config_path = "NULL";  // argv[4];//NULL

  int buffer_size = 5;
  float det_threshold = 0.5;
  bool write_image = true;
  std::string str_image_root(argv[1]);
  std::string str_dst_root = std::string(argv[2]);
  if (!create_directory(str_dst_root)) {
    std::cout << "create directory:" << str_dst_root << " failed\n";
  }
  std::string str_dst_video = join_path(str_dst_root, get_directory_name(str_image_root));
  if (!create_directory(str_dst_video)) {
    std::cout << "create directory:" << str_dst_video << " failed\n";
    // return CVI_FAILURE;
  }
  g_out_dir = str_dst_video;
  if (buffer_size <= 0) {
    printf("buffer size must be larger than 0.\n");
    return CVI_FAILURE;
  }
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 1);
  cviai_handle_t ai_handle = NULL;
  cviai_service_handle_t service_handle = NULL;
  cviai_app_handle_t app_handle = NULL;

  int vpss_grp = 1;
  ret = CVI_AI_CreateHandle2(&ai_handle, vpss_grp, 0);
  ret |= CVI_AI_Service_CreateHandle(&service_handle, ai_handle);
  // ret |= CVI_AI_Service_EnableTPUDraw(service_handle, true);
  ret |= CVI_AI_APP_CreateHandle(&app_handle, ai_handle);
  printf("to facecap init\n");
  ret |= CVI_AI_APP_FaceCapture_Init(app_handle, (uint32_t)buffer_size);
  printf("to quick setup\n");

  cvai_service_feature_array_t feat_gallery;
  memset(&feat_gallery, 0, sizeof(feat_gallery));
  CVI_AI_SUPPORTED_MODEL_E fr_model_id = CVI_AI_SUPPORTED_MODEL_FACERECOGNITION;
  ret |= CVI_AI_APP_FaceCapture_QuickSetUp(app_handle, fd_model_id, fr_model_id, fd_model_path,
                                           NULL, NULL);
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  if (ret != CVI_SUCCESS) {
    release_system(ai_handle, service_handle, app_handle);
    return CVI_FAILURE;
  }

  CVI_AI_SetModelThreshold(ai_handle, fd_model_id, det_threshold);

  app_mode = intelligent;  // APP_MODE_e(atoi(mode_id));
  printf("finish init \n");

  std::vector<std::string> gallery_names;
  bool do_face_recog = false;
  if (ive_handle == NULL) {
    printf("CreateHandle failed with %#x!\n", ret);
    ret = CVI_FAILURE;
  }
  if (do_face_recog) {
    const char *gimg = "/mnt/data/admin1_data/datasets/ivs_eval_set/image/yitong/register.jpg";
    ret = register_gallery_face(app_handle, ive_handle, gimg, &feat_gallery, gallery_names);
    std::cout << "register ret:" << ret << std::endl;
    if (ret == CVI_SUCCESS) {
      std::cout << "to register gallery\n";
      ret = CVI_AI_Service_RegisterFeatureArray(service_handle, feat_gallery, COS_SIMILARITY);
      std::cout << "finish register gallery\n";
    }
  }
  std::cout << "to start:\n";
  CVI_AI_APP_FaceCapture_SetMode(app_handle, CYCLE);

  face_capture_config_t app_cfg;
  CVI_AI_APP_FaceCapture_GetDefaultConfig(&app_cfg);
  if (!strcmp(config_path, "NULL")) {
    printf("Use Default Config...\n");
  }
  if (ret == CVI_FAILURE) {
    release_system(ai_handle, service_handle, app_handle);
    return CVI_FAILURE;
  }
  app_cfg.thr_quality = 0.1;
  app_cfg.thr_quality_high = 0.95;
  app_cfg.thr_size_min = 20;
  app_cfg.miss_time_limit = 20;
  app_cfg.store_RGB888 = true;
  app_cfg.store_feature = true;
  app_cfg.qa_method = 0;
  app_cfg.img_capture_flag = 1;  // capture whole frame
  CVI_AI_APP_FaceCapture_SetConfig(app_handle, &app_cfg);

  memset(&g_obj_meta_0, 0, sizeof(cvai_object_t));
  memset(&g_obj_meta_1, 0, sizeof(cvai_object_t));

  pthread_t io_thread;
  pthread_create(&io_thread, NULL, pImageWrite, NULL);
  const int face_label = 11;

  std::string cap_result = str_dst_video + std::string("/cap_result.log");
  FILE *fp = fopen(cap_result.c_str(), "w");

  int num_append = 0;
  IVE_IMAGE_TYPE_E img_format = IVE_IMAGE_TYPE_U8C3_PACKAGE;  // IVE_IMAGE_TYPE_U8C3_PACKAGE;
  for (int img_idx = 0; img_idx < 1000; img_idx++) {
    std::cout << "processing:" << img_idx << "/530\n";
    char szimg[256];
    sprintf(szimg, "%s/%08d.jpg", str_image_root.c_str(), img_idx);
    std::cout << "processing:" << img_idx << "/1000,path:" << szimg << std::endl;
    bool empty_img = false;
    IVE_IMAGE_S image;
    VIDEO_FRAME_INFO_S fdFrame;
    ret = load_image_file(ive_handle, szimg, image, fdFrame, img_format);
    printf("read image ret:%d width:%d\n", ret, (int)fdFrame.stVFrame.u32Width);
    if (ret != CVI_SUCCESS) {
      if (img_idx < 100) {
        release_system(ai_handle, service_handle, app_handle);
        break;
      }
      printf("load image failed\n");
      empty_img = true;
      ret = load_image_file(ive_handle, "/mnt/data/admin1_data/alios_test/black_1080p.jpg", image,
                            fdFrame, img_format);
      num_append++;
      if (num_append > 30) {
        break;
      }
    }

    if (ret != CVI_SUCCESS) {
      std::cout << "Convert to video frame failed with:" << ret << ",file:" << std::string(szimg)
                << std::endl;

      continue;
    }

    int alive_person_num = COUNT_ALIVE(app_handle->face_cpt_info);
    printf("ALIVE persons: %d\n", alive_person_num);
    ret = CVI_AI_APP_FaceCapture_Run(app_handle, &fdFrame);
    if (ret != CVI_SUCCESS) {
      printf("CVI_AI_APP_FaceCapture_Run failed with %#x\n", ret);
      break;
    }

    if (write_image) {
      std::cout << "to export trackinginfo\n";
      if (!empty_img)
        export_tracking_info(app_handle->face_cpt_info, str_dst_video, img_idx,
                             fdFrame.stVFrame.u32Width, fdFrame.stVFrame.u32Height, face_label);
      std::cout << "to capture trackinginfo\n";
      for (uint32_t i = 0; i < app_handle->face_cpt_info->size; i++) {
        if (!app_handle->face_cpt_info->_output[i]) continue;

        cvai_face_info_t *pface_info = &app_handle->face_cpt_info->data[i].info;
        do_face_match(service_handle, gallery_names, pface_info);
        tracker_state_e state = app_handle->face_cpt_info->data[i].state;
        uint32_t counter = app_handle->face_cpt_info->data[i]._out_counter;
        uint64_t u_id = app_handle->face_cpt_info->data[i].info.unique_id;
        float face_quality = app_handle->face_cpt_info->data[i].info.face_quality;

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
        printf("to export track:%d,frm:%d,capfrm:%d\n", (int)u_id, (int)i,
               (int)app_handle->face_cpt_info->data[i].cap_timestamp);
        /* Copy image data to buffer */
        memset(&data_buffer[target_idx], 0, sizeof(data_buffer[target_idx]));
        data_buffer[target_idx].u_id = u_id;
        data_buffer[target_idx].quality = face_quality;
        data_buffer[target_idx].state = state;
        data_buffer[target_idx].counter = counter;
        data_buffer[target_idx].match_score = pface_info->recog_score;
        data_buffer[target_idx].frame_id = app_handle->face_cpt_info->data[i].cap_timestamp;
        memcpy(data_buffer[target_idx].name, pface_info->name, 128);
        /* NOTE: Make sure the image type is IVE_IMAGE_TYPE_U8C3_PACKAGE */

        CVI_AI_CopyImage(&app_handle->face_cpt_info->data[i].image, &data_buffer[target_idx].image);
        {
          SMT_MutexAutoLock(IOMutex, lock);
          rear_idx = target_idx;
        }
      }
    }
    printf("to release frame\n");
    CVI_AI_ReleaseImage(&fdFrame);
#ifndef CV181X
    CVI_SYS_FreeI(ive_handle, &image);
#endif
  }
  fclose(fp);
  printf("to release system\n");
  CVI_IVE_DestroyHandle(ive_handle);
  bRunImageWriter = false;
  bRunVideoOutput = false;
  pthread_join(io_thread, NULL);

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