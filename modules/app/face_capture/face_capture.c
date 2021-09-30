#include "face_capture.h"
#include "cviai_log.hpp"

#define DEFAULT_SIZE 10
#define QUALITY_THRESHOLD 0.9
#define QUALITY_HIGH_THRESHOLD 0.99
#define MISS_TIME_LIMIT 40
#define FAST_MODE_INTERVAL 100
#define CYCLE_MODE_INTERVAL 20

#define USE_FACE_FEATURE 0

CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                    cvai_tracker_t *tracker_meta);
CVI_S32 clean_data(face_capture_t *face_cpt_info);
CVI_S32 capture_face(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle,
                     VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta);
bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                  float current_quality);
#if USE_FACE_FEATURE
void feature_copy(cvai_feature_t *src_feature, cvai_feature_t *dst_feature);
#endif
int get_alive_num(face_capture_t *face_cpt_info);

// TODO
CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info) {
  // clean heap data
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    printf("[APP::FaceCapture] Free Face Info[%u]\n", j);
    // free(feature);
    // free(face);
    free(&face_cpt_info->data[j]);
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_Init(face_capture_t **face_cpt_info) {
  if (*face_cpt_info != NULL) {
    LOGW("[APP::FaceCapture] already exist.\n");
    return CVIAI_SUCCESS;
  }
  face_capture_t *new_face_cpt_info = (face_capture_t *)malloc(sizeof(face_capture_t));
  memset(new_face_cpt_info, 0, sizeof(face_capture_t));
  new_face_cpt_info->size = DEFAULT_SIZE;
  new_face_cpt_info->data = (face_cpt_data_t *)malloc(sizeof(face_cpt_data_t) * DEFAULT_SIZE);
  memset(new_face_cpt_info->data, 0, sizeof(face_cpt_data_t) * DEFAULT_SIZE);

  new_face_cpt_info->_thr_quality = QUALITY_THRESHOLD;
  new_face_cpt_info->_thr_yaw = 0.5;
  new_face_cpt_info->_thr_pitch = 0.5;
  new_face_cpt_info->_thr_roll = 0.5;

  *face_cpt_info = new_face_cpt_info;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_QuickSetUp(cviai_handle_t ai_handle, const char *fd_model_path,
                                const char *fr_model_path, const char *fq_model_path) {
  CVI_S32 ret = CVIAI_SUCCESS;
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, fd_model_path);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, fr_model_path);
  ret |= CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fq_model_path);
  if (ret != CVIAI_SUCCESS) {
    printf("failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACERECOGNITION, false);

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

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         const IVE_HANDLE ive_handle, VIDEO_FRAME_INFO_S *frame) {
  if (face_cpt_info == NULL) {
    LOGE("[APP::FaceCapture] is not initialized.\n");
    return CVIAI_FAILURE;
  }
  clean_data(face_cpt_info);
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  if (face_cpt_info->last_capture != NULL) {
    free(face_cpt_info->last_capture);
  }

  CVI_AI_RetinaFace(ai_handle, frame, &face_cpt_info->last_faces);
  printf("Found %x faces.\n", face_cpt_info->last_faces.size);
  // CVI_AI_FaceRecognition(ai_handle, frame, &face_cpt_info->last_faces);

  // TODO: optimize FaceQuality (do not inference the faces with bad head pose.)
  CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces);

#if 0
  for (uint32_t j = 0; j < face_cpt_info->last_faces.size; j++) {
    printf("face[%u] quality: %.2f, face post: ( %.2f, %.2f, %.2f)\n", j,
           face_cpt_info->last_faces.info[j].face_quality,
           face_cpt_info->last_faces.info[j].head_pose.yaw,
           face_cpt_info->last_faces.info[j].head_pose.pitch,
           face_cpt_info->last_faces.info[j].head_pose.roll);
  }
#endif

  bool use_DeepSORT = false;
  CVI_AI_DeepSORT_Face(ai_handle, &face_cpt_info->last_faces, &face_cpt_info->last_trackers,
                       use_DeepSORT);

  face_cpt_info->last_capture = (bool *)malloc(sizeof(bool) * face_cpt_info->last_faces.size);
  memset(face_cpt_info->last_capture, 0, sizeof(bool) * face_cpt_info->last_faces.size);
  update_data(face_cpt_info, &face_cpt_info->last_faces, &face_cpt_info->last_trackers);
  capture_face(face_cpt_info, ive_handle, frame, &face_cpt_info->last_faces);

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
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE) {
      // free(feature);
      // free(face);
      memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
    }
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  return CVIAI_SUCCESS;
}

CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                    cvai_tracker_t *tracker_meta) {
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
        if (face_cpt_info->data[j].state == IDLE) {
          printf("[APP::FaceCapture] Create Face Info[%u]\n", j);
          face_cpt_info->data[j].miss_counter = 0;
          face_cpt_info->data[j].state = ALIVE;
          memcpy(&face_cpt_info->data[j].info, &face_meta->info[i], sizeof(cvai_face_info_t));
          /* set useless heap data structure to 0 */
          memset(&face_cpt_info->data[j].info.pts, 0, sizeof(cvai_pts_t));
          memset(&face_cpt_info->data[j].info.feature, 0, sizeof(cvai_feature_t));
#if USE_FACE_FEATURE
          feature_copy(&face_cpt_info->data[j].info.feature, &face_meta->info[i].feature);
#endif
          /* always capture faces in the first frame. */
          face_cpt_info->data[j]._capture = true;
          face_cpt_info->last_capture[i] = true;
          is_created = true;
          break;
        }
      }
      /* if fail to create, return CVIAI_FAILURE */
      if (!is_created) {
        LOGE("buffer overflow.\n");
        return CVIAI_FAILURE;
      }
    } else {
      face_cpt_info->data[match_idx].miss_counter = 0;
      bool capture = false;
      switch (face_cpt_info->mode) {
        case AUTO: {
          if (face_cpt_info->data[match_idx].info.face_quality < QUALITY_HIGH_THRESHOLD) {
            capture = is_qualified(face_cpt_info, &face_meta->info[i],
                                   face_cpt_info->data[match_idx].info.face_quality);
          }
        } break;
        case FAST: {
          // TODO
          if (face_cpt_info->_time - face_cpt_info->data[match_idx]._timestamp <
              FAST_MODE_INTERVAL) {
            capture = is_qualified(face_cpt_info, &face_meta->info[i],
                                   face_cpt_info->data[match_idx].info.face_quality);
            // fake capture for visualization
            face_cpt_info->last_capture[i] = true;
          }
        } break;
        case CYCLE: {
          if (face_cpt_info->_time - face_cpt_info->data[match_idx]._timestamp >
              CYCLE_MODE_INTERVAL) {
            capture = true;
            // capture = is_qualified(face_cpt_info, &face_meta->info[i], 0);
          }
        } break;
        default: {
          LOGE("Unsupported type.\n");
          return CVIAI_ERR_INVALID_ARGS;
        } break;
      }
      /* if found, check whether the quality(or feature) need to be update. */
      if (capture) {
        printf("[APP::FaceCapture] Update Face Info[%u]\n", match_idx);
        memcpy(&face_cpt_info->data[match_idx].info, &face_meta->info[i], sizeof(cvai_face_info_t));
        /* set useless heap data structure to 0 */
        memset(&face_cpt_info->data[match_idx].info.pts, 0, sizeof(cvai_pts_t));
        memset(&face_cpt_info->data[match_idx].info.feature, 0, sizeof(cvai_feature_t));
#if USE_FACE_FEATURE
        feature_copy(&face_cpt_info->data[match_idx].info.feature, &face_meta->info[i].feature);
#endif
        face_cpt_info->data[match_idx]._capture = true;
        face_cpt_info->last_capture[i] = true;
      }
    }
  }

  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE &&
        face_cpt_info->data[j].miss_counter > MISS_TIME_LIMIT) {
      face_cpt_info->data[j].state = MISS;
    }
  }

  return CVIAI_SUCCESS;
}

CVI_S32 clean_data(face_capture_t *face_cpt_info) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == MISS) {
      printf("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      // free(feature);
      // free(face);
      memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 capture_face(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle,
                     VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state != ALIVE || !(face_cpt_info->data[j]._capture)) {
      continue;
    }
    printf("Capture Face[%u]!\n", j);
    face_cpt_info->data[j]._timestamp = face_cpt_info->_time;
    face_cpt_info->data[j]._capture = false;

    // CVI_S32 ret = CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[match_idx].face,
    //                                     &face_meta->info[i]);
    // if (ret != CVIAI_SUCCESS) {
    //   printf("AI get aligned face failed(2).\n");
    //   return false;
    // }
  }
  return CVIAI_SUCCESS;
}

bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                  float current_quality) {
  if (face_info->face_quality >= face_cpt_info->_thr_quality &&
      face_info->face_quality > current_quality &&
      face_info->head_pose.yaw < face_cpt_info->_thr_yaw &&
      face_info->head_pose.pitch < face_cpt_info->_thr_pitch &&
      face_info->head_pose.roll < face_cpt_info->_thr_roll) {
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