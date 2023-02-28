#include "face_capture.h"
#include "default_config.h"

#include <math.h>
#include "core/cviai_utils.h"
#include "cviai_log.hpp"
#include "service/cviai_service.h"

#define ABS(x) ((x) >= 0 ? (x) : (-(x)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define FACE_AREA_STANDARD (112 * 112)
#define EYE_DISTANCE_STANDARD 80.

#define MEMORY_LIMIT (16 * 1024 * 1024) /* example: 16MB */

#define UPDATE_VALUE_MIN 0.1
// TODO: Use cooldown to avoid too much updating
#define UPDATE_COOLDOWN 3
#define CAPTURE_FACE_LIVE_TIME_EXTEND 5

/* face capture functions (core) */
static CVI_S32 update_data(face_capture_t *face_cpt_info, cvai_face_t *face_meta,
                           cvai_tracker_t *tracker_meta);
static CVI_S32 clean_data(face_capture_t *face_cpt_info);
static CVI_S32 capture_face(face_capture_t *face_cpt_info, VIDEO_FRAME_INFO_S *frame,
                            cvai_face_t *face_meta);
static CVI_S32 update_output_state(face_capture_t *face_cpt_info);
static CVI_S32 extract_cropped_face(const cviai_handle_t ai_handle, face_capture_t *face_cpt_info);
static CVI_S32 capture_face_with_vpss(const cviai_handle_t ai_handle, face_capture_t *face_cpt_info,
                                      VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta);
static void face_quality_assessment(VIDEO_FRAME_INFO_S *frame, cvai_face_t *face, bool *skip,
                                    quality_assessment_e qa_method, float thr_laplacian);

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
  LOGI("_FaceCapture_QuickSetUp");
  if (fd_model_id != CVI_AI_SUPPORTED_MODEL_RETINAFACE &&
      fd_model_id != CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION &&
      fd_model_id != CVI_AI_SUPPORTED_MODEL_SCRFDFACE) {
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
  if (ret == CVI_SUCCESS) {
    printf("fd model %s open sucessfull\n", fd_model_path);
  }
  if (fr_model_path != NULL) {
    ret |= CVI_AI_OpenModel(ai_handle, fr_model_id, fr_model_path);
    if (ret == CVI_SUCCESS) {
      printf("fr model %s open sucessfull\n", fr_model_path);
    }
  }

  if (fq_model_path != NULL) {
    ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY, fq_model_path);
  }

#ifndef NO_OPENCV
  printf("enter noopencv\n");
  if (fd_model_id == CVI_AI_SUPPORTED_MODEL_RETINAFACE) {
    face_cpt_info->fd_inference = CVI_AI_RetinaFace;
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  } else if (fd_model_id == CVI_AI_SUPPORTED_MODEL_SCRFDFACE) {
    face_cpt_info->fd_inference = CVI_AI_ScrFDFace;
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, false);
  } else {
    face_cpt_info->fd_inference = CVI_AI_FaceMaskDetection;
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_FACEMASKDETECTION, false);
  }
#else
  if (fd_model_id == CVI_AI_SUPPORTED_MODEL_SCRFDFACE) {
    face_cpt_info->fd_inference = CVI_AI_ScrFDFace;
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_SCRFDFACE, false);
    printf("set fd_inference as CVI_AI_ScrFDFace\n");
  }
#endif

  face_cpt_info->fd_model = fd_model_id;
  printf("frmodelid:%d\n", (int)fr_model_id);
  face_cpt_info->fr_inference = (fr_model_id == CVI_AI_SUPPORTED_MODEL_FACERECOGNITION)
                                    ? CVI_AI_FaceRecognition
                                    : CVI_AI_FaceAttribute;

  if (ret != CVIAI_SUCCESS) {
    printf("_FaceCapture_QuickSetUp failed with %#x!\n", ret);
    return ret;
  }

  face_cpt_info->use_FQNet = fq_model_path != NULL;
  if (fr_model_path != NULL) {
    face_cpt_info->fr_flag = 1;
  }

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
  printf("CVI_AI_DeepSORT_SetConfig done\n");
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_GetDefaultConfig(face_capture_config_t *cfg) {
  cfg->miss_time_limit = MISS_TIME_LIMIT;
  cfg->thr_size_min = SIZE_MIN_THRESHOLD;
  cfg->thr_size_max = SIZE_MAX_THRESHOLD;
  cfg->qa_method = QUALITY_ASSESSMENT_METHOD;
  cfg->thr_quality = QUALITY_THRESHOLD;
  cfg->thr_quality_high = QUALITY_HIGH_THRESHOLD;
  cfg->thr_yaw = 0.7;
  cfg->thr_pitch = 0.5;
  cfg->thr_roll = 0.65;
  cfg->thr_laplacian = 100;
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
  CVI_AI_DeepSORT_SetConfig(ai_handle, &deepsort_conf, -1, true);
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
       face_cpt_info->fr_flag, face_cpt_info->use_FQNet);
  CVI_S32 ret;
  ret = clean_data(face_cpt_info);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] clean data failed.\n");
    return CVIAI_FAILURE;
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  /* set output signal to 0. */
  for (int i = 0; i < face_cpt_info->size; i++) {
    if (face_cpt_info->_output[i]) {
      face_cpt_info->data[i].info.face_quality =
          0;  // set fq to zero ,force to update data at next time
    }
  }
  memset(face_cpt_info->_output, 0, sizeof(bool) * face_cpt_info->size);

  if (CVIAI_SUCCESS != face_cpt_info->fd_inference(ai_handle, frame, &face_cpt_info->last_faces)) {
    return CVIAI_FAILURE;
  }
  if (face_cpt_info->fr_flag == 1) {
    if (CVI_SUCCESS != face_cpt_info->fr_inference(ai_handle, frame, &face_cpt_info->last_faces)) {
      return CVIAI_FAILURE;
    }
  }

  CVI_AI_Service_FaceAngleForAll(&face_cpt_info->last_faces);
  bool *skip = (bool *)malloc(sizeof(bool) * face_cpt_info->last_faces.size);
  set_skipFQsignal(face_cpt_info, &face_cpt_info->last_faces, skip);
  face_quality_assessment(frame, &face_cpt_info->last_faces, skip, face_cpt_info->cfg.qa_method,
                          face_cpt_info->cfg.thr_laplacian);
  if (face_cpt_info->use_FQNet) {
#ifndef NO_OPENCV
    if (CVIAI_SUCCESS != CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces, skip)) {
      return CVIAI_FAILURE;
    }
#endif
  } else {
    for (uint32_t i = 0; i < face_cpt_info->last_faces.size; i++) {
      // use pose score as face_quality
      face_cpt_info->last_faces.info[i].face_quality = face_cpt_info->last_faces.info[i].pose_score;
    }
  }
  free(skip);

#ifdef DEBUG_CAPTURE
  for (uint32_t j = 0; j < face_cpt_info->last_faces.size; j++) {
    LOGI("face[%u]: quality[%.4f], pose[%.2f][%.2f][%.2f]\n", j,
         face_cpt_info->last_faces.info[j].face_quality,
         face_cpt_info->last_faces.info[j].head_pose.yaw,
         face_cpt_info->last_faces.info[j].head_pose.pitch,
         face_cpt_info->last_faces.info[j].head_pose.roll);
  }
#endif

  CVI_AI_DeepSORT_Face(ai_handle, &face_cpt_info->last_faces, &face_cpt_info->last_trackers);

  ret = update_data(face_cpt_info, &face_cpt_info->last_faces, &face_cpt_info->last_trackers);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] update face failed.\n");
    return CVIAI_FAILURE;
  }
  bool use_vpss = true;
  if (use_vpss) {
    ret = capture_face_with_vpss(ai_handle, face_cpt_info, frame, &face_cpt_info->last_faces);
  } else {
    ret = capture_face(face_cpt_info, frame, &face_cpt_info->last_faces);
  }
  update_output_state(face_cpt_info);

  if (face_cpt_info->fr_flag == 2) {
    // extract face feature from cropped image
    extract_cropped_face(ai_handle, face_cpt_info);
  }
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::FaceCapture] capture face failed.\n");
    return CVIAI_FAILURE;
  }

#ifdef DEBUG_CAPTURE
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

static float get_score(cvai_bbox_t *bbox, cvai_pts_t *pts_info, cvai_head_pose_t *pose) {
  float nose_x = pts_info->x[2];
  // float nose_y = pts_info->y[2];
  float left_max = MIN(pts_info->x[0], pts_info->x[3]);
  float right_max = MAX(pts_info->x[1], pts_info->x[4]);
  float width = bbox->x2 - bbox->x1;
  float height = bbox->y2 - bbox->y1;
  float l_ = nose_x - left_max;
  float r_ = right_max - nose_x;
  // printf("box:[%.3f,%.3f,%.3f,%.3f],w:%.3f,h:%.3f\n",bbox->x1,bbox->y1,bbox->x2,bbox->y2,width,height);
  // printf("kpts=[");
  // for(int i = 0; i < 5; i++){
  //   printf("%.3f,%.3f,",pts_info->x[i],pts_info->y[i]);
  // }

  float eye_diff_x = pts_info->x[1] - pts_info->x[0];
  float eye_diff_y = pts_info->y[1] - pts_info->y[0];
  float eye_size = sqrt(eye_diff_x * eye_diff_x + eye_diff_y * eye_diff_y);

  float mouth_diff_x = pts_info->x[4] - pts_info->x[3];
  float mouth_diff_y = pts_info->y[4] - pts_info->y[3];
  float mouth_size = sqrt(mouth_diff_x * mouth_diff_x + mouth_diff_y * mouth_diff_y);

  if (pts_info->x[1] > bbox->x2 || pts_info->x[2] > bbox->x2 || pts_info->x[4] > bbox->x2 ||
      pts_info->x[0] < bbox->x1 || pts_info->x[2] < bbox->x1 || pts_info->x[3] < bbox->x1) {
    return 0.0;
  } else if ((l_ + 0.01 * width) < 0 || (r_ + 0.01 * width) < 0 || (eye_size / width) < 0.25 ||
             (mouth_size / width) < 0.15) {
    return 0.0;
  } else if ((pts_info->y[0] < bbox->y1 || pts_info->y[1] < bbox->y1 || pts_info->y[3] > bbox->y2 ||
              pts_info->y[4] > bbox->y2)) {
    return 0.0;
  } else if (width * height < (25 * 25)) {
    return 0.0;
  } else {
    float face_size = ((bbox->y2 - bbox->y1) + (bbox->x2 - bbox->x1)) / 2;
    float area_score = (face_size - 64) / 128.0;
    if (area_score > 1.5) area_score = 1.5;
    float size_score = 0;
    float pose_score = 1. - (ABS(pose->yaw) + ABS(pose->pitch) + ABS(pose->roll) * 0.5) / 3.;
    float wpose = 0.6;
    float warea = 0.4;
    if (face_size > 64) {
      wpose = 0.8;
      warea = 0.2;
      size_score = eye_size / (bbox->x2 - bbox->x1);
      size_score += mouth_size / (bbox->x2 - bbox->x1);
    }

    pose_score = pose_score * wpose + 0.2 * size_score + area_score * warea;
    return pose_score;
  }
}

static void face_quality_assessment(VIDEO_FRAME_INFO_S *frame, cvai_face_t *face, bool *skip,
                                    quality_assessment_e qa_method, float thr_laplacian) {
  /* NOTE: Make sure the coordinate is recovered by RetinaFace */
  for (uint32_t i = 0; i < face->size; i++) {
    if (skip != NULL && skip[i]) {
      face->info[i].pose_score = 0;
      LOGD("skip face:%u\n", i);
      continue;
    }
    if (qa_method == AREA_RATIO) {
      cvai_bbox_t *bbox = &face->info[i].bbox;
      cvai_pts_t *pts_info = &face->info[i].pts;
      cvai_head_pose_t *pose = &face->info[i].head_pose;
      face->info[i].pose_score = get_score(bbox, pts_info, pose);
      LOGD("face posescore:%.3f\n", face->info[i].pose_score);
    } else if (qa_method == EYES_DISTANCE) {
      float dx = face->info[i].pts.x[0] - face->info[i].pts.x[1];
      float dy = face->info[i].pts.y[0] - face->info[i].pts.y[1];
      float dist_score = sqrt(dx * dx + dy * dy) / EYE_DISTANCE_STANDARD;
      face->info[i].pose_score = (dist_score >= 1.) ? 1. : dist_score;
    } else if (qa_method == LAPLACIAN) {
      static const float face_area = 112 * 112;
      static const float laplacian_threshold = 8.0; /* tune this value for different condition */
      float score;
      CVI_AI_Face_Quality_Laplacian(frame, &face->info[i], &score);
      score /= face_area;
      score /= laplacian_threshold;
      if (score > 1.0) score = 1.0;
      face->info[i].pose_score = score;
    } else if (qa_method == MIX) {
      cvai_bbox_t *bbox = &face->info[i].bbox;
      cvai_pts_t *pts_info = &face->info[i].pts;
      cvai_head_pose_t *pose = &face->info[i].head_pose;
      float score1, score2;
      score1 = get_score(bbox, pts_info, pose);
      face->info[i].pose_score = score1;
      if (score1 >= 0.1) {
        float face_area = (bbox->y2 - bbox->y1) * (bbox->x2 - bbox->x1);
        float laplacian_threshold =
            thr_laplacian; /* tune this value for different condition default:100 */
        CVI_AI_Face_Quality_Laplacian(frame, &face->info[i], &score2);
        // score2 = sqrt(score2);
        float area_score = MIN(1.0, face_area / (90 * 90));
        score2 = score2 * area_score;
        if (score2 < laplacian_threshold) face->info[i].pose_score = 0.0;
        face->info[i].sharpness_score = score2;
      }
    }
  }
  return;
}
static CVI_S32 update_output_state(face_capture_t *face_cpt_info) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    // only process alive track and good face quality face
    if (face_cpt_info->data[j].state != ALIVE) {
      continue;
    }
    if (face_cpt_info->data[j].miss_counter >
        face_cpt_info->cfg.miss_time_limit + CAPTURE_FACE_LIVE_TIME_EXTEND) {
      face_cpt_info->data[j].state = MISS;
      if (face_cpt_info->data[j].info.face_quality < face_cpt_info->cfg.thr_quality) continue;

      if (face_cpt_info->mode == AUTO) {
        face_cpt_info->_output[j] = true;
      } else if ((face_cpt_info->mode == FAST || face_cpt_info->mode == CYCLE) &&
                 face_cpt_info->data[j]._out_counter == 0) {
        face_cpt_info->_output[j] = true;
      }
    } else if (face_cpt_info->data[j].info.face_quality >= face_cpt_info->cfg.thr_quality) {
      int _time = face_cpt_info->_time - face_cpt_info->data[j]._timestamp;
      if (_time == 0) continue;

      if (face_cpt_info->mode == AUTO) {
        if (face_cpt_info->cfg.auto_m_fast_cap && _time >= face_cpt_info->cfg.fast_m_interval &&
            face_cpt_info->data[j]._out_counter < 1) {
          face_cpt_info->_output[j] = true;
          face_cpt_info->data[j]._out_counter += 1;
        }
        if (face_cpt_info->cfg.auto_m_time_limit != 0 &&
            _time == face_cpt_info->cfg.auto_m_time_limit) {
          /* Time's up */
          face_cpt_info->_output[j] = true;
          face_cpt_info->data[j]._out_counter += 1;
        }
      } else if (face_cpt_info->mode == FAST &&
                 face_cpt_info->data[j]._out_counter < face_cpt_info->cfg.fast_m_capture_num &&
                 _time == face_cpt_info->cfg.fast_m_interval - 1) {
        face_cpt_info->_output[j] = true;
        face_cpt_info->data[j]._out_counter += 1;
      } else if (face_cpt_info->mode == CYCLE && _time == face_cpt_info->cfg.cycle_m_interval - 1) {
        face_cpt_info->_output[j] = true;
        face_cpt_info->data[j]._out_counter += 1;
        LOGD("update output flag,interval:%u\n", face_cpt_info->cfg.cycle_m_interval);
      }
    }
    if (face_cpt_info->_output[j]) {
      face_cpt_info->data[j]._timestamp = face_cpt_info->_time;  // update output timestamp
    }
  }
  return CVI_SUCCESS;
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
    // if is low quality face,do not generate capture data
    bool toskip = face_meta->info[i].face_quality < face_cpt_info->cfg.thr_quality ||
                  face_meta->info[i].pose_score == 0;
    uint64_t trk_id = face_meta->info[i].unique_id;
    LOGD("update_data,trackid:%d,quality:%.3f,pscore:%.3f\n", (int)trk_id,
         face_meta->info[i].face_quality, face_meta->info[i].pose_score);

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
      if (toskip) {
        LOGD("update_data,skip to generate capture data,trackid:%d,pscore:%f\n",
             (int)face_meta->info[i].unique_id, face_meta->info[i].pose_score);
        continue;
      }
      /* if not found, create new one. */
      bool is_created = false;
      /* search available index for new tracker. */
      for (uint32_t j = 0; j < face_cpt_info->size; j++) {
        if (face_cpt_info->data[j].state == IDLE && face_cpt_info->data[j]._capture == false) {
          LOGI("[APP::FaceCapture] Create Face Info[%u],trackid:%d\n", j, (int)trk_id);
          // uint32_t tid = face_cpt_info->data[j].info.unique_id;
          // uint32_t frmid = face_cpt_info->_time;
          face_cpt_info->data[j].miss_counter = 0;
          memcpy(&face_cpt_info->data[j].info, &face_meta->info[i], sizeof(cvai_face_info_t));

          /* copy face feature */
          uint32_t feature_size =
              getFeatureTypeSize(face_meta->info[i].feature.type) * face_meta->info[i].feature.size;
          if (face_cpt_info->cfg.store_feature && feature_size > 0) {
            if (feature_size != getFeatureTypeSize(face_cpt_info->data[j].info.feature.type) *
                                    face_cpt_info->data[j].info.feature.size) {
              free(face_cpt_info->data[j].info.feature.ptr);
              face_cpt_info->data[j].info.feature.ptr = (int8_t *)malloc(feature_size);
            }
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
          face_cpt_info->data[j].cap_timestamp = face_cpt_info->_time;
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
      // int _time = face_cpt_info->_time - face_cpt_info->data[match_idx]._timestamp;
      float current_quality = face_cpt_info->data[match_idx].info.face_quality;
      capture = is_qualified(face_cpt_info, &face_meta->info[i], current_quality + 0.03);
      /* if found, check whether the quality(or feature) need to be update. */
      if (capture) {
        LOGI("[APP::FaceCapture] Update Face Info[%u]\n", match_idx);
        int8_t *p_feature = face_cpt_info->data[match_idx].info.feature.ptr;
        float *p_pts_x = face_cpt_info->data[match_idx].info.pts.x;
        float *p_pts_y = face_cpt_info->data[match_idx].info.pts.y;
        // store matched name
        memcpy(face_meta->info[i].name, face_cpt_info->data[match_idx].info.name,
               sizeof(face_cpt_info->data[match_idx].info.name));
        memcpy(&face_cpt_info->data[match_idx].info, &face_meta->info[i], sizeof(cvai_face_info_t));
        face_cpt_info->data[match_idx].info.feature.ptr = p_feature;
        face_cpt_info->data[match_idx].info.pts.x = p_pts_x;
        face_cpt_info->data[match_idx].info.pts.y = p_pts_y;

        /* copy face feature */
        uint32_t feature_size =
            getFeatureTypeSize(face_meta->info[i].feature.type) * face_meta->info[i].feature.size;
        if (face_cpt_info->cfg.store_feature && feature_size > 0) {
          if (feature_size != getFeatureTypeSize(face_cpt_info->data[match_idx].info.feature.type) *
                                  face_cpt_info->data[match_idx].info.feature.size) {
            free(face_cpt_info->data[match_idx].info.feature.ptr);
            face_cpt_info->data[match_idx].info.feature.ptr = (int8_t *)malloc(feature_size);
          }
          memcpy(face_cpt_info->data[match_idx].info.feature.ptr, face_meta->info[i].feature.ptr,
                 feature_size);
        }

        /* copy face 5 landmarks */
        memcpy(face_cpt_info->data[match_idx].info.pts.x, face_meta->info[i].pts.x,
               sizeof(float) * 5);
        memcpy(face_cpt_info->data[match_idx].info.pts.y, face_meta->info[i].pts.y,
               sizeof(float) * 5);

        face_cpt_info->data[match_idx]._capture = true;
        face_cpt_info->data[match_idx].cap_timestamp = face_cpt_info->_time;
      }
      // update matched name
      memcpy(face_meta->info[i].name, face_cpt_info->data[match_idx].info.name,
             sizeof(face_cpt_info->data[match_idx].info.name));
      LOGI("update_data,update matched trackid:%d,name:%s\n", (int)face_meta->info[i].unique_id,
           face_meta->info[i].name);
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
int update_extend_resize_info(const float frame_width, const float frame_height,
                              cvai_face_info_t *face_info, cvai_bbox_t *p_dst_box) {
  cvai_bbox_t bbox = face_info->bbox;
  int w_pad = (bbox.x2 - bbox.x1) * 0.2;
  int h_pad = (bbox.y2 - bbox.y1) * 0.2;

  // bbox new coordinate after extern
  float x1 = bbox.x1 - w_pad;
  float y1 = bbox.y1 - h_pad;
  float x2 = bbox.x2 + w_pad;
  float y2 = bbox.y2 + h_pad;

  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = x1 > 0 ? x1 : 0;
  new_bbox.y1 = y1 > 0 ? y1 : 0;
  new_bbox.x2 = x2 < frame_width ? x2 : frame_width - 1;
  new_bbox.y2 = y2 < frame_height ? y2 : frame_height - 1;

  // float ratio, pad_width, pad_height;
  float box_height = new_bbox.y2 - new_bbox.y1;
  float box_width = new_bbox.x2 - new_bbox.x1;

  int mean_hw = (box_width + box_height) / 2;
  int dst_hw = 128;
  if (mean_hw > dst_hw) {
    dst_hw = 256;
  }
  *p_dst_box = new_bbox;
  float ratio_h = dst_hw / box_height;
  float ratio_w = dst_hw / box_width;
  for (uint32_t j = 0; j < face_info->pts.size; ++j) {
    face_info->pts.x[j] = (face_info->pts.x[j] - new_bbox.x1) * ratio_w;
    face_info->pts.y[j] = (face_info->pts.y[j] - new_bbox.y1) * ratio_h;
  }
  face_info->bbox.x1 = (face_info->bbox.x1 - new_bbox.x1) * ratio_w;
  face_info->bbox.y1 = (face_info->bbox.y1 - new_bbox.y1) * ratio_h;
  face_info->bbox.x2 = (face_info->bbox.x2 - new_bbox.x1) * ratio_w;
  face_info->bbox.y2 = (face_info->bbox.y2 - new_bbox.y1) * ratio_h;

  return dst_hw;
}
static CVI_S32 capture_face_with_vpss(const cviai_handle_t ai_handle, face_capture_t *face_cpt_info,
                                      VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta) {
  LOGI("[APP::FaceCapture] Capture Face\n");
  int ret = CVIAI_SUCCESS;
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
    cvai_bbox_t newbox;
    int dst_size = update_extend_resize_info(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                             &face_cpt_info->data[j].info, &newbox);
    if (face_cpt_info->data[j].image.width != dst_size ||
        face_cpt_info->data[j].image.height != dst_size) {
      CVI_AI_Free(&face_cpt_info->data[j].image);
      CVI_AI_CreateImage(&face_cpt_info->data[j].image, dst_size, dst_size, PIXEL_FORMAT_RGB_888);
    }
    ret = CVI_AI_CropImage_With_VPSS(ai_handle, face_cpt_info->fd_model, frame, &newbox,
                                     &face_cpt_info->data[j].image);
    if (ret != CVIAI_SUCCESS) {
      LOGW("error crop image,modelid:%d\n", (int)face_cpt_info->fd_model);
    } else {
      LOGD("update cropped image,width:%u,step:%u,trackid:%d\n", face_cpt_info->data[j].image.width,
           face_cpt_info->data[j].image.stride[0], (int)face_cpt_info->data[j].info.unique_id);
    }

    face_cpt_info->data[j]._capture = false;
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
    printf("Pixel format [%d] is not supported.\n", frame->stVFrame.enPixelFormat);
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
        printf("FaceData[%u] state[%s], h[%u], w[%u], size[%" PRIu64 "],name:%s\n", i,
               (state == ALIVE) ? "ALIVE" : "MISS", face_cpt_info->data[i].image.height,
               face_cpt_info->data[i].image.width, m, face_cpt_info->data[i].info.name);
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

static CVI_S32 extract_cropped_face(const cviai_handle_t ai_handle, face_capture_t *face_cpt_info) {
  for (uint32_t i = 0; i < face_cpt_info->size; i++) {
    if (face_cpt_info->_output[i]) {
      int ret = CVI_AI_FaceFeatureExtract(
          ai_handle, face_cpt_info->data[i].image.pix[0], face_cpt_info->data[i].image.width,
          face_cpt_info->data[i].image.height, face_cpt_info->data[i].image.stride[0],
          &face_cpt_info->data[i].info);
      LOGI("extract face feature,trackid:%d,ret:%d\n", (int)face_cpt_info->data[i].info.unique_id,
           ret);
    }
  }
  return CVI_SUCCESS;
}
