#include "face_capture.h"
#include <math.h>
#include "cviai_log.hpp"
#include "service/cviai_service.h"

#define ABS(x) ((x) >= 0 ? (x) : (-(x)))

#define SIZE_THRESHOLD 32
#define QUALITY_THRESHOLD 0.9
#define QUALITY_HIGH_THRESHOLD 0.99
#define MISS_TIME_LIMIT 40
#define FAST_MODE_INTERVAL 40
#define FAST_MODE_CAPTURE_NUM 3
#define CYCLE_MODE_INTERVAL 20
#define AUTO_MODE_INTERVAL 20

#define USE_FACE_FEATURE 0
#define WRITE_FACE_IN

CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                    cvai_tracker_t *tracker_meta);
CVI_S32 clean_data(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle);
CVI_S32 capture_face(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle,
                     VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta);

void set_skipFQsignal(face_capture_t *face_cpt_info, cvai_face_t *face_info, bool *skip);
bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                  float current_quality);
#if USE_FACE_FEATURE
void feature_copy(cvai_feature_t *src_feature, cvai_feature_t *dst_feature);
#endif
int get_alive_num(face_capture_t *face_cpt_info);
void show_config(face_capture_config_t *cfg);
CVI_S32 get_ive_image_type(PIXEL_FORMAT_E enPixelFormat, IVE_IMAGE_TYPE_E *enType);

CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle) {
  // LOGI("[APP::FaceCapture] Free FaceCapture Data\n");
  // clean heap data
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].face_image.tpu_block != NULL) {
      CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
    }
    CVI_AI_Free(&face_cpt_info->data[j].info);
  }
  free(face_cpt_info->data);
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  free(face_cpt_info->_output);

  free(face_cpt_info);
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_Init(face_capture_t **face_cpt_info, uint32_t buffer_size) {
  if (*face_cpt_info != NULL) {
    LOGW("[APP::FaceCapture] already exist.\n");
    return CVIAI_SUCCESS;
  }
  LOGI("[APP::FaceCapture] Initialize (Buffer Size: %u)\n", buffer_size);
  face_capture_t *new_face_cpt_info = (face_capture_t *)malloc(sizeof(face_capture_t));
  memset(new_face_cpt_info, 0, sizeof(face_capture_t));
  new_face_cpt_info->size = buffer_size;
  new_face_cpt_info->data = (face_cpt_data_t *)malloc(sizeof(face_cpt_data_t) * buffer_size);
  memset(new_face_cpt_info->data, 0, sizeof(face_cpt_data_t) * buffer_size);

  new_face_cpt_info->_output = (bool *)malloc(sizeof(bool) * buffer_size);
  memset(new_face_cpt_info->_output, 0, sizeof(bool) * buffer_size);

  _FaceCapture_GetDefaultConfig(&new_face_cpt_info->cfg);

  *face_cpt_info = new_face_cpt_info;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_QuickSetUp(cviai_handle_t ai_handle, const char *fd_model_path,
                                const char *fq_model_path) {
  CVI_S32 ret = CVIAI_SUCCESS;
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_model_path);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fq_model_path);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  /* Init DeepSORT */
  CVI_AI_DeepSORT_Init(ai_handle, false);
#if 1
  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.ktracker_conf.max_unmatched_num = 20;
  ds_conf.ktracker_conf.accreditation_threshold = 10;
  ds_conf.ktracker_conf.P_std_beta[2] = 0.1;
  ds_conf.ktracker_conf.P_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.Q_std_beta[2] = 0.1;
  ds_conf.kfilter_conf.Q_std_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.R_std_beta[2] = 0.1;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);
#endif

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_GetDefaultConfig(face_capture_config_t *cfg) {
  cfg->miss_time_limit = MISS_TIME_LIMIT;
  cfg->thr_size = 32;
  cfg->thr_quality = QUALITY_THRESHOLD;
  cfg->thr_quality_high = QUALITY_HIGH_THRESHOLD;
  cfg->thr_yaw = 0.5;
  cfg->thr_pitch = 0.5;
  cfg->thr_roll = 0.5;
  cfg->fast_m_interval = FAST_MODE_INTERVAL;
  cfg->fast_m_capture_num = FAST_MODE_CAPTURE_NUM;
  cfg->cycle_m_interval = CYCLE_MODE_INTERVAL;

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_SetConfig(face_capture_t *face_cpt_info, face_capture_config_t *cfg) {
  memcpy(&face_cpt_info->cfg, cfg, sizeof(face_capture_config_t));
  show_config(&face_cpt_info->cfg);
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         const IVE_HANDLE ive_handle, VIDEO_FRAME_INFO_S *frame) {
  if (face_cpt_info == NULL) {
    LOGE("[APP::FaceCapture] is not initialized.\n");
    return CVIAI_FAILURE;
  }
  CVI_S32 ret;
  ret = clean_data(face_cpt_info, ive_handle);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] clean data failed.\n");
    return CVI_FAILURE;
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  /* set output signal to 0. */
  memset(face_cpt_info->_output, 0, sizeof(bool) * face_cpt_info->size);

  CVI_AI_RetinaFace(ai_handle, frame, &face_cpt_info->last_faces);
  printf("Found %x faces.\n", face_cpt_info->last_faces.size);
  // CVI_AI_FaceRecognition(ai_handle, frame, &face_cpt_info->last_faces);

  // TODO: optimize FaceQuality (do not inference the faces with bad head pose.)
  CVI_AI_Service_FaceAngleForAll(&face_cpt_info->last_faces);
  bool *skip = (bool *)malloc(sizeof(bool) * face_cpt_info->last_faces.size);
  set_skipFQsignal(face_cpt_info, &face_cpt_info->last_faces, skip);
  CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces, skip);
  free(skip);

#if 1
  for (uint32_t j = 0; j < face_cpt_info->last_faces.size; j++) {
    printf("face[%u] quality: %.2f, pose: ( %.2f, %.2f, %.2f)\n", j,
           face_cpt_info->last_faces.info[j].face_quality,
           face_cpt_info->last_faces.info[j].head_pose.yaw,
           face_cpt_info->last_faces.info[j].head_pose.pitch,
           face_cpt_info->last_faces.info[j].head_pose.roll);
  }
#endif

  bool use_DeepSORT = false;
  CVI_AI_DeepSORT_Face(ai_handle, &face_cpt_info->last_faces, &face_cpt_info->last_trackers,
                       use_DeepSORT);

  ret = update_data(face_cpt_info, &face_cpt_info->last_faces, &face_cpt_info->last_trackers);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] update face failed.\n");
    return CVI_FAILURE;
  }
  ret = capture_face(face_cpt_info, ive_handle, frame, &face_cpt_info->last_faces);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] capture face failed.\n");
    return CVI_FAILURE;
  }

  /* update timestamp*/
  face_cpt_info->_time =
      (face_cpt_info->_time == 0xffffffffffffffff) ? 0 : face_cpt_info->_time + 1;

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_SetMode(face_capture_t *face_cpt_info, capture_mode_e mode) {
  face_cpt_info->mode = mode;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_CleanAll(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state != IDLE) {
      LOGI("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      CVI_S32 ret = CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
      if (ret != CVI_SUCCESS) {
        LOGE("CVI_SYS_FreeI fail with %d\n", ret);
        return ret;
      }
      CVI_AI_Free(&face_cpt_info->data[j].info);
      // memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
      face_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                    cvai_tracker_t *tracker_meta) {
  // LOGI("[APP::FaceCapture] Update Data\n");
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      face_cpt_info->data[j].miss_counter += 1;
    }
  }
  for (uint32_t i = 0; i < tracker_meta->size; i++) {
    /* we only consider the stable tracker in this sample code. */
    if (tracker_meta->info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }
    // TODO: Skip the faces whoes size is too small...

    uint64_t trk_id = face_meta->info[i].unique_id;
    /* check whether the tracker id exist or not. */
    int match_idx = -1;
    for (uint32_t j = 0; j < face_cpt_info->size; j++) {
      if (face_cpt_info->data[j].state == ALIVE &&
          face_cpt_info->data[j].info.unique_id == trk_id) {
        match_idx = (int)j;
        break;
      }
    }
    if (match_idx == -1) {
      /* if not found, create new one. */
      bool is_created = false;
      /* search available index for new tracker. */
      for (uint32_t j = 0; j < face_cpt_info->size; j++) {
        if (face_cpt_info->data[j].state == IDLE && face_cpt_info->data[j]._capture == false) {
          LOGI("[APP::FaceCapture] Create Face Info[%u]\n", j);
          face_cpt_info->data[j].miss_counter = 0;
          memcpy(&face_cpt_info->data[j].info, &face_meta->info[i], sizeof(cvai_face_info_t));
          /* set useless heap data structure to 0 */
          memset(&face_cpt_info->data[j].info.pts, 0, sizeof(cvai_pts_t));
          memset(&face_cpt_info->data[j].info.feature, 0, sizeof(cvai_feature_t));
#if USE_FACE_FEATURE
          feature_copy(&face_cpt_info->data[j].info.feature, &face_meta->info[i].feature);
#endif
          /* always capture faces in the first frame. */
          face_cpt_info->data[j]._capture = true;
          face_cpt_info->data[j]._timestamp = face_cpt_info->_time;
          is_created = true;
          break;
        }
      }
      /* fail to create */
      if (!is_created) {
        LOGW("[APP::FaceCapture] Buffer overflow! (Ignore face[%u])\n", i);
      }
    } else {
      face_cpt_info->data[match_idx].miss_counter = 0;
      bool capture = false;
      uint64_t _time = face_cpt_info->_time - face_cpt_info->data[match_idx]._timestamp;
      switch (face_cpt_info->mode) {
        case AUTO: {
          if (face_cpt_info->data[match_idx].info.face_quality <
              face_cpt_info->cfg.thr_quality_high) {
            capture = is_qualified(face_cpt_info, &face_meta->info[i],
                                   face_cpt_info->data[match_idx].info.face_quality);
          }
        } break;
        case FAST: {
          if (face_cpt_info->data[match_idx]._out_counter < face_cpt_info->cfg.fast_m_capture_num) {
            if (_time < face_cpt_info->cfg.fast_m_interval) {
              capture = is_qualified(face_cpt_info, &face_meta->info[i],
                                     face_cpt_info->data[match_idx].info.face_quality);
            } else {
              capture = true;
              face_cpt_info->data[match_idx]._timestamp = face_cpt_info->_time;
            }
          }
        } break;
        case CYCLE: {
          if (_time < face_cpt_info->cfg.cycle_m_interval) {
            capture = is_qualified(face_cpt_info, &face_meta->info[i],
                                   face_cpt_info->data[match_idx].info.face_quality);
            // capture = true;
          } else {
            capture = true;
            face_cpt_info->data[match_idx]._timestamp = face_cpt_info->_time;
          }
        } break;
        default: {
          // TODO: consider non-free tracker because it won't set MISS
          LOGE("Unsupported type.\n");
          return CVIAI_ERR_INVALID_ARGS;
        } break;
      }
      /* if found, check whether the quality(or feature) need to be update. */
      if (capture) {
        LOGI("[APP::FaceCapture] Update Face Info[%u]\n", match_idx);
        memcpy(&face_cpt_info->data[match_idx].info, &face_meta->info[i], sizeof(cvai_face_info_t));
        /* set useless heap data structure to 0 */
        memset(&face_cpt_info->data[match_idx].info.pts, 0, sizeof(cvai_pts_t));
        memset(&face_cpt_info->data[match_idx].info.feature, 0, sizeof(cvai_feature_t));
#if USE_FACE_FEATURE
        feature_copy(&face_cpt_info->data[match_idx].info.feature, &face_meta->info[i].feature);
#endif
        face_cpt_info->data[match_idx]._capture = true;
      }
      switch (face_cpt_info->mode) {
        case AUTO: {
          if (_time >= AUTO_MODE_INTERVAL && face_cpt_info->data[match_idx]._out_counter < 1 &&
              is_qualified(face_cpt_info, &face_cpt_info->data[match_idx].info, -1)) {
            face_cpt_info->_output[match_idx] = true;
            face_cpt_info->data[match_idx]._out_counter += 1;
          }
        } break;
        case FAST: {
          if (face_cpt_info->data[match_idx]._out_counter < face_cpt_info->cfg.fast_m_capture_num) {
            if (_time == face_cpt_info->cfg.fast_m_interval - 1 &&
                is_qualified(face_cpt_info, &face_cpt_info->data[match_idx].info, -1)) {
              face_cpt_info->_output[match_idx] = true;
              face_cpt_info->data[match_idx]._out_counter += 1;
            }
          }
        } break;
        case CYCLE: {
          if (_time == face_cpt_info->cfg.cycle_m_interval - 1 &&
              is_qualified(face_cpt_info, &face_cpt_info->data[match_idx].info, -1)) {
            face_cpt_info->_output[match_idx] = true;
            face_cpt_info->data[match_idx]._out_counter += 1;
          }
        } break;
        default:
          return CVIAI_FAILURE;
      }
    }
  }

  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE &&
        face_cpt_info->data[j].miss_counter > face_cpt_info->cfg.miss_time_limit) {
      face_cpt_info->data[j].state = MISS;
      if (face_cpt_info->mode == AUTO) {
        face_cpt_info->_output[j] = true;
      }
    }
  }

  return CVIAI_SUCCESS;
}

CVI_S32 clean_data(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == MISS) {
      LOGI("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      CVI_S32 ret = CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
      if (ret != CVI_SUCCESS) {
        LOGE("CVI_SYS_FreeI fail with %d\n", ret);
        return ret;
      }
      CVI_AI_Free(&face_cpt_info->data[j].info);
      // memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
      face_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 capture_face(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle,
                     VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta) {
  // LOGI("[APP::FaceCapture] Capture Face\n");
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }

  bool capture = false;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    // if (face_cpt_info->data[j].state == ALIVE && face_cpt_info->data[j]._capture) {
    if (face_cpt_info->data[j]._capture) {
      capture = true;
      break;
    }
  }
  if (!capture) {
    return CVIAI_SUCCESS;
  }
  IVE_IMAGE_S ive_frame;
  bool do_unmap = false;
  size_t image_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    do_unmap = true;
  }
  CVI_S32 ret = CVI_IVE_VideoFrameInfo2Image(frame, &ive_frame);
  if (ret != CVI_SUCCESS) {
    LOGE("Convert to video frame failed with %#x!\n", ret);
  }

  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (!(face_cpt_info->data[j]._capture)) {
      continue;
    }
    bool first_capture = false;
    if (face_cpt_info->data[j].state == ALIVE) {
      CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
    } else {
      /* first capture */
      face_cpt_info->data[j].state = ALIVE;
      first_capture = true;
    }
    LOGI("Capture Face[%u] (%s)!\n", j, (first_capture) ? "INIT" : "UPDATE");

    /* CVI_IVE_SubImage not support PIXEL_FORMAT_RGB_888 */
    CVI_U16 x1 = (CVI_U16)roundf(face_cpt_info->data[j].info.bbox.x1);
    CVI_U16 y1 = (CVI_U16)roundf(face_cpt_info->data[j].info.bbox.y1);
    CVI_U16 x2 = (CVI_U16)roundf(face_cpt_info->data[j].info.bbox.x2);
    CVI_U16 y2 = (CVI_U16)roundf(face_cpt_info->data[j].info.bbox.y2);
    CVI_U16 h = y2 - y1 + 1;
    CVI_U16 w = x2 - x1 + 1;
    // printf("Crop (h: %hu,w: %hu) [ %hu, %hu, %hu, %hu]\n", h, w, x1, y1, x2, y2);

    IVE_IMAGE_TYPE_E enType;
    ret = get_ive_image_type(frame->stVFrame.enPixelFormat, &enType);
    if (ret != CVI_SUCCESS) {
      printf("Get IVE IMAGE TYPE Failed!\n");
      CVI_SYS_FreeI(ive_handle, &ive_frame);
      return ret;
    }

    // LOGI("[DEBUG] CVI_IVE_CreateImage [ %hu, %hu] (BEGIN)\n", h, w);
    ret =
        CVI_IVE_CreateImage(ive_handle, &face_cpt_info->data[j].face_image, ive_frame.enType, w, h);
    // LOGI("[DEBUG] CVI_IVE_CreateImage (END)\n");
    if (ret != CVIAI_SUCCESS) {
      LOGE("Create IVE IMAGE Failed!\n");
      CVI_SYS_FreeI(ive_handle, &ive_frame);
      return ret;
    }
    CVI_U16 stride_face = face_cpt_info->data[j].face_image.u16Stride[0];
    CVI_U16 stride_frame = ive_frame.u16Stride[0];
    size_t cpy_size = (size_t)w * 3 * sizeof(CVI_U8);

    CVI_U16 t = 0;
    for (CVI_U16 i = y1; i <= y2; i++) {
      memcpy(face_cpt_info->data[j].face_image.pu8VirAddr[0] + t * stride_face,
             ive_frame.pu8VirAddr[0] + i * stride_frame + x1 * 3, cpy_size);
      t += 1;
    }

    face_cpt_info->data[j]._capture = false;
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], image_size);
  }
  CVI_SYS_FreeI(ive_handle, &ive_frame);
  return CVIAI_SUCCESS;
}

void set_skipFQsignal(face_capture_t *face_cpt_info, cvai_face_t *face_meta, bool *skip) {
  // TODO: OPTIMIZE
  memset(skip, 0, sizeof(bool) * face_meta->size);
  bool care_size = face_cpt_info->cfg.thr_size != -1;
  for (uint32_t i = 0; i < face_meta->size; i++) {
    if (care_size) {
      float h = face_meta->info[i].bbox.y2 - face_meta->info[i].bbox.y1;
      float w = face_meta->info[i].bbox.x2 - face_meta->info[i].bbox.x1;
      if (h < (float)face_cpt_info->cfg.thr_size || w < (float)face_cpt_info->cfg.thr_size) {
        skip[i] = true;
        continue;
      }
    }
    if (ABS(face_meta->info[i].head_pose.yaw) > face_cpt_info->cfg.thr_yaw ||
        ABS(face_meta->info[i].head_pose.pitch) > face_cpt_info->cfg.thr_pitch ||
        ABS(face_meta->info[i].head_pose.roll) > face_cpt_info->cfg.thr_roll) {
      skip[i] = true;
    }
  }
}

bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                  float current_quality) {
  if (face_info->face_quality >= face_cpt_info->cfg.thr_quality &&
      face_info->face_quality > current_quality) {
    return true;
  }
  return false;
}

#if USE_FACE_FEATURE
// TODO: check correctness
void feature_copy(cvai_feature_t *src_feature, cvai_feature_t *dst_feature) {
  dst_feature->size = src_feature->size;
  dst_feature->type = src_feature->type;
  size_t type_size = getFeatureTypeSize(dst_feature->type);
  dst_feature->ptr = (int8_t *)malloc(dst_feature->size * type_size);
  memcpy(dst_feature->ptr, src_feature->ptr, dst_feature->size * type_size);
}
#endif

int get_alive_num(face_capture_t *face_cpt_info) {
  int counter = 0;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      counter += 1;
    }
  }
  return counter;
}

void show_config(face_capture_config_t *cfg) {
  printf("@@@ Face Capture Config @@@\n");
  printf(" - Miss Time Limit:   : %u\n", cfg->miss_time_limit);
  printf(" - Thr Size           : %i\n", cfg->thr_size);
  printf(" - Thr Quality        : %.2f\n", cfg->thr_quality);
  printf(" - Thr Quality (High) : %.2f\n", cfg->thr_quality_high);
  printf(" - Thr Yaw    : %.2f\n", cfg->thr_yaw);
  printf(" - Thr Pitch  : %.2f\n", cfg->thr_pitch);
  printf(" - Thr Roll   : %.2f\n", cfg->thr_roll);
  printf("[Fast] Interval: %u\n", cfg->fast_m_interval);
  printf("[Fast] Capture Num: %u\n", cfg->fast_m_capture_num);
  printf("[Cycle] Interval: %u\n\n", cfg->cycle_m_interval);
  return;
}

CVI_S32 get_ive_image_type(PIXEL_FORMAT_E enPixelFormat, IVE_IMAGE_TYPE_E *enType) {
  // printf("enPixelFormat = %d\n", enPixelFormat);
  switch (enPixelFormat) {
    case PIXEL_FORMAT_YUV_400:
      *enType = IVE_IMAGE_TYPE_U8C1;
      return CVI_SUCCESS;
    case PIXEL_FORMAT_YUV_PLANAR_420:
      *enType = IVE_IMAGE_TYPE_YUV420P;
      return CVI_SUCCESS;
    case PIXEL_FORMAT_YUV_PLANAR_422:
      *enType = IVE_IMAGE_TYPE_YUV422P;
      return CVI_SUCCESS;
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888:
      *enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
      return CVI_SUCCESS;
    case PIXEL_FORMAT_RGB_888_PLANAR:
      *enType = IVE_IMAGE_TYPE_U8C3_PLANAR;
      return CVI_SUCCESS;
    default:
      LOGE("Unsupported conversion type: %u.\n", enPixelFormat);
      return CVIAI_ERR_INVALID_ARGS;
  }
}
