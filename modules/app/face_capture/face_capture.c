#include "face_capture.h"
#include "default_config.h"

#include <math.h>
#include "core/cviai_utils.h"
#include "cviai_log.hpp"
#include "service/cviai_service.h"

#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define FACE_AREA_STANDARD (112 * 112)
#define EYE_DISTANCE_STANDARD 80.

#define MEMORY_LIMIT (16 * 1024 * 1024) /* example: 16MB */

#define UPDATE_VALUE_MIN 0.1
// TODO: Use cooldown to avoid too much updating
#define UPDATE_COOLDOWN 3
#define CAPTURE_FACE_LIVE_TIME_EXTEND 5

#define USE_FACE_FEATURE 0

/* face capture functions (core) */
static CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                           cvai_tracker_t *tracker_meta);
static CVI_S32 clean_data(face_capture_t *face_cpt_info);
static CVI_S32 capture_face(face_capture_t *face_cpt_info, VIDEO_FRAME_INFO_S *frame,
                            cvai_face_t *face_meta);
static void face_quality_assessment(VIDEO_FRAME_INFO_S *frame, cvai_face_t *face, bool *skip,
                                    quality_assessment_e qa_method);

/* face capture functions (helper) */
static void set_skipFQsignal(face_capture_t *face_cpt_info, cvai_face_t *face_info, bool *skip);
static bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                         float current_quality);

/* other helper functions */
static bool IS_MEMORY_ENOUGH(uint32_t mem_limit, uint64_t mem_used, cvai_image_t *current_image,
                             cvai_bbox_t *new_bbox, PIXEL_FORMAT_E fmt);
static void SUMMARY(face_capture_t *face_cpt_info, uint64_t *size, bool show_detail);
static void SHOW_CONFIG(face_capture_config_t *cfg);

CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info) {
  LOGI("[APP::FaceCapture] Free FaceCapture Data\n");
  if (face_cpt_info != NULL) {
    _FaceCapture_CleanAll(face_cpt_info);

    free(face_cpt_info->data);
    CVI_AI_Free(&face_cpt_info->last_faces);
    CVI_AI_Free(&face_cpt_info->last_trackers);
    free(face_cpt_info->_output);
    free(face_cpt_info);
  }
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
  new_face_cpt_info->_m_limit = MEMORY_LIMIT;

  *face_cpt_info = new_face_cpt_info;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_QuickSetUp(cviai_handle_t ai_handle, face_capture_t *face_cpt_info,
                                int fd_model_id, int fr_model_id, const char *fd_model_path,
                                const char *fr_model_path, const char *fq_model_path) {
  if (fd_model_id != CVI_AI_SUPPORTED_MODEL_RETINAFACE &&
      fd_model_id != CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION) {
    LOGE("invalid face detection model id %d", fd_model_id);
    return CVI_FAILURE;
  }
  if (fr_model_id != CVI_AI_SUPPORTED_MODEL_FACERECOGNITION &&
      fr_model_id != CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE) {
    LOGE("invalid face recognition model id %d", fr_model_id);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret |= CVI_AI_OpenModel(ai_handle, fd_model_id, fd_model_path);
  if (fr_model_path != NULL) {
    ret |= CVI_AI_OpenModel(ai_handle, fr_model_id, fr_model_path);
  }
  if (fq_model_path != NULL) {
    ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fq_model_path);
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  face_cpt_info->fd_inference = (fd_model_id == CVI_AI_SUPPORTED_MODEL_RETINAFACE)
                                    ? CVI_AI_RetinaFace
                                    : CVI_AI_FaceMaskDetection;
  face_cpt_info->fr_inference = (fr_model_id == CVI_AI_SUPPORTED_MODEL_FACERECOGNITION)
                                    ? CVI_AI_FaceRecognition
                                    : CVI_AI_FaceAttribute;

  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  face_cpt_info->do_FR = fr_model_path != NULL;
  face_cpt_info->use_FQNet = fq_model_path != NULL;

  /* Init DeepSORT */
  CVI_AI_DeepSORT_Init(ai_handle, false);

  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  ds_conf.ktracker_conf.max_unmatched_num = 20;
  ds_conf.ktracker_conf.accreditation_threshold = 10;
  ds_conf.ktracker_conf.P_beta[2] = 0.1;
  ds_conf.ktracker_conf.P_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.Q_beta[2] = 0.1;
  ds_conf.kfilter_conf.Q_beta[6] = 2.5e-2;
  ds_conf.kfilter_conf.R_beta[2] = 0.1;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_GetDefaultConfig(face_capture_config_t *cfg) {
  cfg->miss_time_limit = MISS_TIME_LIMIT;
  cfg->thr_size_min = SIZE_MIN_THRESHOLD;
  cfg->thr_size_max = SIZE_MAX_THRESHOLD;
  cfg->qa_method = QUALITY_ASSESSMENT_METHOD;
  cfg->thr_quality = QUALITY_THRESHOLD;
  cfg->thr_quality_high = QUALITY_HIGH_THRESHOLD;
  cfg->thr_yaw = 0.5;
  cfg->thr_pitch = 0.5;
  cfg->thr_roll = 0.5;
  cfg->fast_m_interval = FAST_MODE_INTERVAL;
  cfg->fast_m_capture_num = FAST_MODE_CAPTURE_NUM;
  cfg->cycle_m_interval = CYCLE_MODE_INTERVAL;
  cfg->auto_m_time_limit = AUTO_MODE_TIME_LIMIT;
  cfg->auto_m_fast_cap = true;
  cfg->capture_aligned_face = false;
  cfg->capture_extended_face = false;
  cfg->store_RGB888 = false;

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_SetConfig(face_capture_t *face_cpt_info, face_capture_config_t *cfg,
                               cviai_handle_t ai_handle) {
  memcpy(&face_cpt_info->cfg, cfg, sizeof(face_capture_config_t));
  if (face_cpt_info->cfg.capture_aligned_face && face_cpt_info->cfg.capture_extended_face) {
    LOGW("set capture_extended_face = false because capture_aligned_face is true.");
    face_cpt_info->cfg.capture_extended_face = false;
  }
  cvai_deepsort_config_t deepsort_conf;
  CVI_AI_DeepSORT_GetConfig(ai_handle, &deepsort_conf, -1);
  deepsort_conf.ktracker_conf.max_unmatched_num = cfg->miss_time_limit;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &deepsort_conf, -1, false);
  SHOW_CONFIG(&face_cpt_info->cfg);
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         VIDEO_FRAME_INFO_S *frame) {
  if (face_cpt_info == NULL) {
    LOGE("[APP::FaceCapture] is not initialized.\n");
    return CVIAI_FAILURE;
  }
  LOGI("[APP::FaceCapture] RUN (MODE: %d, FR: %d, FQ: %d)\n", face_cpt_info->mode,
       face_cpt_info->do_FR, face_cpt_info->use_FQNet);
  CVI_S32 ret;
  ret = clean_data(face_cpt_info);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] clean data failed.\n");
    return CVIAI_FAILURE;
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  /* set output signal to 0. */
  memset(face_cpt_info->_output, 0, sizeof(bool) * face_cpt_info->size);

  if (CVIAI_SUCCESS != face_cpt_info->fd_inference(ai_handle, frame, &face_cpt_info->last_faces)) {
    return CVIAI_FAILURE;
  }
  if (face_cpt_info->do_FR) {
    if (CVI_SUCCESS != face_cpt_info->fr_inference(ai_handle, frame, &face_cpt_info->last_faces)) {
      return CVIAI_FAILURE;
    }
  }

  CVI_AI_Service_FaceAngleForAll(&face_cpt_info->last_faces);
  bool *skip = (bool *)malloc(sizeof(bool) * face_cpt_info->last_faces.size);
  set_skipFQsignal(face_cpt_info, &face_cpt_info->last_faces, skip);
  if (face_cpt_info->use_FQNet) {
    if (CVIAI_SUCCESS != CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces, skip)) {
      return CVIAI_FAILURE;
    }
  } else {
    face_quality_assessment(frame, &face_cpt_info->last_faces, NULL, face_cpt_info->cfg.qa_method);
  }
  free(skip);

#if 0
  for (uint32_t j = 0; j < face_cpt_info->last_faces.size; j++) {
    LOGI("face[%u]: quality[%.4f], pose[%.2f][%.2f][%.2f]\n", j,
         face_cpt_info->last_faces.info[j].face_quality,
         face_cpt_info->last_faces.info[j].head_pose.yaw,
         face_cpt_info->last_faces.info[j].head_pose.pitch,
         face_cpt_info->last_faces.info[j].head_pose.roll);
  }
#endif

  bool use_DeepSORT = face_cpt_info->do_FR;
  CVI_AI_DeepSORT_Face(ai_handle, &face_cpt_info->last_faces, &face_cpt_info->last_trackers,
                       use_DeepSORT);

  ret = update_data(face_cpt_info, &face_cpt_info->last_faces, &face_cpt_info->last_trackers);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] update face failed.\n");
    return CVIAI_FAILURE;
  }
  ret = capture_face(face_cpt_info, frame, &face_cpt_info->last_faces);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] capture face failed.\n");
    return CVIAI_FAILURE;
  }

#if 0
  uint64_t mem_used;
  SUMMARY(face_cpt_info, &mem_used, true);
#endif

  /* update timestamp*/
  face_cpt_info->_time =
      (face_cpt_info->_time == 0xffffffffffffffff) ? 0 : face_cpt_info->_time + 1;

  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_SetMode(face_capture_t *face_cpt_info, capture_mode_e mode) {
  face_cpt_info->mode = mode;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_CleanAll(face_capture_t *face_cpt_info) {
  /* Release tracking data */
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state != IDLE) {
      LOGI("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      CVI_AI_Free(&face_cpt_info->data[j].image);
      CVI_AI_Free(&face_cpt_info->data[j].info);
      face_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

static void face_quality_assessment(VIDEO_FRAME_INFO_S *frame, cvai_face_t *face, bool *skip,
                                    quality_assessment_e qa_method) {
  /* NOTE: Make sure the coordinate is recovered by RetinaFace */
  for (uint32_t i = 0; i < face->size; i++) {
    if (skip != NULL && skip[i]) {
      continue;
    }
    switch (qa_method) {
      case AREA_RATIO: {
        cvai_bbox_t *bbox = &face->info[i].bbox;
        float face_area = (bbox->y2 - bbox->y1) * (bbox->x2 - bbox->x1);
        float area_score = MIN(1.0, face_area / FACE_AREA_STANDARD);
        cvai_head_pose_t *pose = &face->info[i].head_pose;
        float pose_score = 1. - (ABS(pose->yaw) + ABS(pose->pitch) + ABS(pose->roll)) / 3.;
        // face->info[i].face_quality = (area_score + pose_score) / 2.;
        face->info[i].face_quality = area_score * pose_score;
      } break;
      case EYES_DISTANCE: {
        float dx = face->info[i].pts.x[0] - face->info[i].pts.x[1];
        float dy = face->info[i].pts.y[0] - face->info[i].pts.y[1];
        float dist_score = sqrt(dx * dx + dy * dy) / EYE_DISTANCE_STANDARD;
        face->info[i].face_quality = (dist_score >= 1.) ? 1. : dist_score;
      } break;
      case LAPLACIAN: {
        static const float face_area = 112 * 112;
        static const float laplacian_threshold = 8.0; /* tune this value for different condition */
        float score;
        CVI_AI_Face_Quality_Laplacian(frame, &face->info[i], &score);
        score /= face_area;
        score /= laplacian_threshold;
        if (score > 1.0) score = 1.0;
        face->info[i].face_quality = score;
      } break;
      default: {
        LOGE("Unknown QA method.\n");
      } break;
    }
  }
  return;
}

static CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                           cvai_tracker_t *tracker_meta) {
  LOGI("[APP::FaceCapture] Update Data\n");
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
          // LOGI("[APP::FaceCapture] Create Face Info[%u]\n", j);
          face_cpt_info->data[j].miss_counter = 0;
          memcpy(&face_cpt_info->data[j].info, &face_meta->info[i], sizeof(cvai_face_info_t));

          /* copy face feature */
          if (face_cpt_info->do_FR == false || face_cpt_info->cfg.store_feature == false) {
            memset(&face_cpt_info->data[j].info.feature, 0, sizeof(cvai_feature_t));
          } else {
            uint32_t feature_size = getFeatureTypeSize(face_meta->info[i].feature.type) *
                                    face_meta->info[i].feature.size;
            face_cpt_info->data[j].info.feature.ptr = (int8_t *)malloc(feature_size);
            memcpy(face_cpt_info->data[j].info.feature.ptr, face_meta->info[i].feature.ptr,
                   feature_size);
          }

          /* copy face 5 landmarks */
          face_cpt_info->data[j].info.pts.size = 5;
          face_cpt_info->data[j].info.pts.x = (float *)malloc(sizeof(float) * 5);
          face_cpt_info->data[j].info.pts.y = (float *)malloc(sizeof(float) * 5);
          memcpy(face_cpt_info->data[j].info.pts.x, face_meta->info[i].pts.x, sizeof(float) * 5);
          memcpy(face_cpt_info->data[j].info.pts.y, face_meta->info[i].pts.y, sizeof(float) * 5);

          /* always capture faces in the first frame. */
          face_cpt_info->data[j]._capture = true;
          face_cpt_info->data[j]._timestamp = face_cpt_info->_time;
          face_cpt_info->data[j]._out_counter = 0;
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
      float current_quality = face_cpt_info->data[match_idx].info.face_quality;
      switch (face_cpt_info->mode) {
        case AUTO: {
          bool time_out = (face_cpt_info->cfg.auto_m_time_limit != 0) &&
                          (_time > face_cpt_info->cfg.auto_m_time_limit);
          if (!time_out && current_quality < face_cpt_info->cfg.thr_quality_high) {
            float update_quality_threshold =
                (current_quality < face_cpt_info->cfg.thr_quality)
                    ? 0.
                    : MIN(face_cpt_info->cfg.thr_quality_high, current_quality + UPDATE_VALUE_MIN);
            capture = is_qualified(face_cpt_info, &face_meta->info[i], update_quality_threshold);
          }
        } break;
        case FAST: {
          if (face_cpt_info->data[match_idx]._out_counter < face_cpt_info->cfg.fast_m_capture_num) {
            if (_time < face_cpt_info->cfg.fast_m_interval) {
              float update_quality_threshold;
              if (current_quality < face_cpt_info->cfg.thr_quality) {
                update_quality_threshold = -1;
              } else {
                update_quality_threshold = ((current_quality + UPDATE_VALUE_MIN) >= 1.0)
                                               ? current_quality
                                               : current_quality + UPDATE_VALUE_MIN;
              }
              capture = is_qualified(face_cpt_info, &face_meta->info[i], update_quality_threshold);
            } else {
              capture = true;
              face_cpt_info->data[match_idx]._timestamp = face_cpt_info->_time;
            }
          }
        } break;
        case CYCLE: {
          if (_time < face_cpt_info->cfg.cycle_m_interval) {
            float update_quality_threshold;
            if (current_quality < face_cpt_info->cfg.thr_quality) {
              update_quality_threshold = -1;
            } else {
              update_quality_threshold = ((current_quality + UPDATE_VALUE_MIN) >= 1.0)
                                             ? current_quality
                                             : current_quality + UPDATE_VALUE_MIN;
            }
            capture = is_qualified(face_cpt_info, &face_meta->info[i], update_quality_threshold);
          } else {
            capture = true;
            face_cpt_info->data[match_idx]._timestamp = face_cpt_info->_time;
          }
        } break;
        default: {
          // NOTE: consider non-free tracker because it won't set MISS
          LOGE("Unsupported type.\n");
          return CVIAI_ERR_INVALID_ARGS;
        } break;
      }
      /* if found, check whether the quality(or feature) need to be update. */
      if (capture) {
        LOGI("[APP::FaceCapture] Update Face Info[%u]\n", match_idx);
        int8_t *p_feature = face_cpt_info->data[match_idx].info.feature.ptr;
        float *p_pts_x = face_cpt_info->data[match_idx].info.pts.x;
        float *p_pts_y = face_cpt_info->data[match_idx].info.pts.y;
        memcpy(&face_cpt_info->data[match_idx].info, &face_meta->info[i], sizeof(cvai_face_info_t));
        face_cpt_info->data[match_idx].info.feature.ptr = p_feature;
        face_cpt_info->data[match_idx].info.pts.x = p_pts_x;
        face_cpt_info->data[match_idx].info.pts.y = p_pts_y;

        /* copy face feature */
        if (face_cpt_info->do_FR == false || face_cpt_info->cfg.store_feature == false) {
          memset(&face_cpt_info->data[match_idx].info.feature, 0, sizeof(cvai_feature_t));
        } else {
          uint32_t feature_size =
              getFeatureTypeSize(face_meta->info[i].feature.type) * face_meta->info[i].feature.size;
          memcpy(face_cpt_info->data[match_idx].info.feature.ptr, face_meta->info[i].feature.ptr,
                 feature_size);
        }

        /* copy face 5 landmarks */
        memcpy(face_cpt_info->data[match_idx].info.pts.x, face_meta->info[i].pts.x,
               sizeof(float) * 5);
        memcpy(face_cpt_info->data[match_idx].info.pts.y, face_meta->info[i].pts.y,
               sizeof(float) * 5);

        face_cpt_info->data[match_idx]._capture = true;
      }
      switch (face_cpt_info->mode) {
        case AUTO: {
          /* We use fast interval for the first capture */
          if (face_cpt_info->cfg.auto_m_fast_cap) {
            if (_time >= face_cpt_info->cfg.fast_m_interval &&
                face_cpt_info->data[match_idx]._out_counter < 1 &&
                is_qualified(face_cpt_info, &face_cpt_info->data[match_idx].info, -1)) {
              face_cpt_info->_output[match_idx] = true;
              face_cpt_info->data[match_idx]._out_counter += 1;
            }
          }
          if (face_cpt_info->cfg.auto_m_time_limit != 0) {
            /* Time's up */
            if (_time == face_cpt_info->cfg.auto_m_time_limit &&
                is_qualified(face_cpt_info, &face_cpt_info->data[match_idx].info, -1)) {
              face_cpt_info->_output[match_idx] = true;
              face_cpt_info->data[match_idx]._out_counter += 1;
            }
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
    /* NOTE: For more flexible application, we do not remove the tracker immediately when time out
     */
    if (face_cpt_info->data[j].state == ALIVE &&
        face_cpt_info->data[j].miss_counter >
            face_cpt_info->cfg.miss_time_limit + CAPTURE_FACE_LIVE_TIME_EXTEND) {
      face_cpt_info->data[j].state = MISS;
      switch (face_cpt_info->mode) {
        case AUTO: {
          if (is_qualified(face_cpt_info, &face_cpt_info->data[j].info, -1)) {
            face_cpt_info->_output[j] = true;
          }
        } break;
        case FAST:
        case CYCLE: {
          /* at least capture 1 face */
          if (is_qualified(face_cpt_info, &face_cpt_info->data[j].info, -1) &&
              face_cpt_info->data[j]._out_counter == 0) {
            face_cpt_info->_output[j] = true;
          }
        } break;
        default:
          return CVIAI_FAILURE;
          break;
      }
    }
  }

  return CVIAI_SUCCESS;
}

static CVI_S32 clean_data(face_capture_t *face_cpt_info) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == MISS) {
      LOGI("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      CVI_AI_Free(&face_cpt_info->data[j].image);
      CVI_AI_Free(&face_cpt_info->data[j].info);
      face_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

static CVI_S32 capture_face(face_capture_t *face_cpt_info, VIDEO_FRAME_INFO_S *frame,
                            cvai_face_t *face_meta) {
  LOGI("[APP::FaceCapture] Capture Face\n");
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE("Pixel format [%d] is not supported.\n", frame->stVFrame.enPixelFormat);
    return CVIAI_ERR_INVALID_ARGS;
  }

  bool capture = false;
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j]._capture) {
      capture = true;
      break;
    }
  }
  if (!capture) {
    return CVIAI_SUCCESS;
  }
  /* Estimate memory used */
  uint64_t mem_used;
  SUMMARY(face_cpt_info, &mem_used, false);

  bool do_unmap = false;
  size_t image_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
    do_unmap = true;
  }

  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (!(face_cpt_info->data[j]._capture)) {
      continue;
    }
    bool first_capture = false;
    if (face_cpt_info->data[j].state != ALIVE) {
      /* first capture */
      face_cpt_info->data[j].state = ALIVE;
      first_capture = true;
    }
    LOGI("Capture Face[%u] (%s)!\n", j, (first_capture) ? "INIT" : "UPDATE");

    /* Check remaining memory space */
    if (!IS_MEMORY_ENOUGH(face_cpt_info->_m_limit, mem_used, &face_cpt_info->data[j].image,
                          &face_cpt_info->data[j].info.bbox, frame->stVFrame.enPixelFormat)) {
      LOGW("Memory is not enough. (drop)\n");
      if (first_capture) {
        face_cpt_info->data[j].state = IDLE;
      }
      continue;
    }
    CVI_AI_Free(&face_cpt_info->data[j].image);

    if (!face_cpt_info->cfg.capture_extended_face) {
      CVI_AI_CropImage_Face(frame, &face_cpt_info->data[j].image, &face_cpt_info->data[j].info,
                            face_cpt_info->cfg.capture_aligned_face,
                            face_cpt_info->cfg.store_RGB888);
    } else {
      float dummy_x, dummy_y;
      CVI_AI_CropImage_Exten(frame, &face_cpt_info->data[j].image,
                             &face_cpt_info->data[j].info.bbox, face_cpt_info->cfg.store_RGB888,
                             0.5, &dummy_x, &dummy_y);
    }

    face_cpt_info->data[j]._capture = false;
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return CVIAI_SUCCESS;
}

static void set_skipFQsignal(face_capture_t *face_cpt_info, cvai_face_t *face_meta, bool *skip) {
  memset(skip, 0, sizeof(bool) * face_meta->size);
  bool care_size_min = face_cpt_info->cfg.thr_size_min != -1;
  bool care_size_max = face_cpt_info->cfg.thr_size_max != -1;
  if (!care_size_min && !care_size_max) return;

  for (uint32_t i = 0; i < face_meta->size; i++) {
    float h = face_meta->info[i].bbox.y2 - face_meta->info[i].bbox.y1;
    float w = face_meta->info[i].bbox.x2 - face_meta->info[i].bbox.x1;
    if (care_size_min) {
      if (h < (float)face_cpt_info->cfg.thr_size_min ||
          w < (float)face_cpt_info->cfg.thr_size_min) {
        skip[i] = true;
        continue;
      }
    }
    if (care_size_max) {
      if (h > (float)face_cpt_info->cfg.thr_size_max ||
          w > (float)face_cpt_info->cfg.thr_size_max) {
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

static bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                         float current_quality) {
  if (face_info->face_quality >= face_cpt_info->cfg.thr_quality &&
      face_info->face_quality > current_quality) {
    return true;
  }
  return false;
}

static bool IS_MEMORY_ENOUGH(uint32_t mem_limit, uint64_t mem_used, cvai_image_t *current_image,
                             cvai_bbox_t *new_bbox, PIXEL_FORMAT_E fmt) {
  mem_used -= (current_image->length[0] + current_image->length[1] + current_image->length[2]);
  uint32_t new_h = (uint32_t)roundf(new_bbox->y2) - (uint32_t)roundf(new_bbox->y1) + 1;
  uint32_t new_w = (uint32_t)roundf(new_bbox->x2) - (uint32_t)roundf(new_bbox->x1) + 1;
  uint64_t new_size;
  CVI_AI_EstimateImageSize(&new_size, ((new_h + 1) >> 1) << 1, ((new_w + 1) >> 1) << 1, fmt);
  if (mem_limit - mem_used < new_size) {
    return false;
  } else {
    // printf("<remaining: %u, needed: %u>\n", mem_limit - mem_used, new_s * new_h);
    return true;
  }
}

static void SUMMARY(face_capture_t *face_cpt_info, uint64_t *size, bool show_detail) {
  *size = 0;
  if (show_detail) {
    printf("@@@@ SUMMARY @@@@\n");
  }
  for (uint32_t i = 0; i < face_cpt_info->size; i++) {
    tracker_state_e state = face_cpt_info->data[i].state;
    if (state == IDLE) {
      if (show_detail) {
        printf("FaceData[%u] state[IDLE]\n", i);
      }
    } else {
      uint64_t m = face_cpt_info->data[i].image.length[0];
      m += face_cpt_info->data[i].image.length[1];
      m += face_cpt_info->data[i].image.length[2];
      if (show_detail) {
        printf("FaceData[%u] state[%s], h[%u], w[%u], size[%" PRIu64 "]\n", i,
               (state == ALIVE) ? "ALIVE" : "MISS", face_cpt_info->data[i].image.height,
               face_cpt_info->data[i].image.width, m);
      }
      *size += m;
    }
  }
  if (show_detail) {
    printf("MEMORY USED: %" PRIu64 "\n\n", *size);
  }
}

static void SHOW_CONFIG(face_capture_config_t *cfg) {
  printf("@@@ Face Capture Config @@@\n");
  printf(" - Miss Time Limit:   : %u\n", cfg->miss_time_limit);
  printf(" - Thr Size (Min)     : %i\n", cfg->thr_size_min);
  printf(" - Thr Size (Max)     : %i\n", cfg->thr_size_max);
  printf(" - Quality Assessment Method : %i\n", cfg->qa_method);
  printf(" - Thr Quality        : %.2f\n", cfg->thr_quality);
  printf(" - Thr Quality (High) : %.2f\n", cfg->thr_quality_high);
  printf(" - Thr Yaw    : %.2f\n", cfg->thr_yaw);
  printf(" - Thr Pitch  : %.2f\n", cfg->thr_pitch);
  printf(" - Thr Roll   : %.2f\n", cfg->thr_roll);
  printf("[Fast] Interval     : %u\n", cfg->fast_m_interval);
  printf("[Fast] Capture Num  : %u\n", cfg->fast_m_capture_num);
  printf("[Cycle] Interval    : %u\n", cfg->cycle_m_interval);
  printf("[Auto] Time Limit   : %u\n\n", cfg->auto_m_time_limit);
  printf("[Auto] Fast Capture : %s\n\n", cfg->auto_m_fast_cap ? "True" : "False");
  printf(" - Capture Aligned Face : %s\n\n", cfg->capture_aligned_face ? "True" : "False");
  printf(" - Store Face Feature   : %s\n\n", cfg->store_feature ? "True" : "False");
  printf(" - Store RGB888         : %s\n\n", cfg->store_RGB888 ? "True" : "False");
  return;
}
