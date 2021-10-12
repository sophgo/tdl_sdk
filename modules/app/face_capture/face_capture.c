#include "face_capture.h"
#include <math.h>
#include "cviai_log.hpp"
#include "service/cviai_service.h"

#define ABS(x) ((x) >= 0 ? (x) : (-(x)))

#define DEFAULT_SIZE 10
#define QUALITY_THRESHOLD 0.9
#define QUALITY_HIGH_THRESHOLD 0.99
#define MISS_TIME_LIMIT 40
#define FAST_MODE_INTERVAL 100
#define CYCLE_MODE_INTERVAL 20

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
CVI_S32 get_ive_image_type(PIXEL_FORMAT_E enPixelFormat, IVE_IMAGE_TYPE_E *enType);

// TODO
CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle) {
  // clean heap data
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    printf("[APP::FaceCapture] Free Face Info[%u]\n", j);
    // free(feature);
    CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
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
  clean_data(face_cpt_info, ive_handle);
  CVI_AI_Free(&face_cpt_info->last_faces);
  CVI_AI_Free(&face_cpt_info->last_trackers);
  if (face_cpt_info->last_capture != NULL) {
    free(face_cpt_info->last_capture);
  }

  CVI_AI_RetinaFace(ai_handle, frame, &face_cpt_info->last_faces);
  printf("Found %x faces.\n", face_cpt_info->last_faces.size);
  // CVI_AI_FaceRecognition(ai_handle, frame, &face_cpt_info->last_faces);

  // TODO: optimize FaceQuality (do not inference the faces with bad head pose.)
  CVI_AI_Service_FaceAngleForAll(&face_cpt_info->last_faces);
  bool *skip = (bool *)malloc(sizeof(bool) * face_cpt_info->last_faces.size);
  set_skipFQsignal(face_cpt_info, &face_cpt_info->last_faces, skip);
  CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces, skip);

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

CVI_S32 clean_data(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == MISS) {
      printf("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      // free(feature);
      CVI_S32 ret = CVI_SYS_FreeI(ive_handle, &face_cpt_info->data[j].face_image);
      if (ret != CVI_SUCCESS) {
        printf("CVI_SYS_FreeI fail with %d\n", ret);
      }
      memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 capture_face(face_capture_t *face_cpt_info, const IVE_HANDLE ive_handle,
                     VIDEO_FRAME_INFO_S *frame, cvai_face_t *face_meta) {
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
    printf("Capture Face[%u] (%s)!\n", j, (first_capture) ? "INIT" : "UPDATE");

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
    if (ret != CVIAI_SUCCESS) {
      printf("Get IVE IMAGE TYPE Failed!\n");
      CVI_SYS_FreeI(ive_handle, &ive_frame);
      return ret;
    }

    ret =
        CVI_IVE_CreateImage(ive_handle, &face_cpt_info->data[j].face_image, ive_frame.enType, w, h);
    if (ret != CVIAI_SUCCESS) {
      printf("Create IVE IMAGE Failed!\n");
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

#ifdef WRITE_FACE_IN
    if (first_capture) {
      char *filename = calloc(32, sizeof(char));
      sprintf(filename, "face_%" PRIu64 "_in.png", face_cpt_info->data[j].info.unique_id);
      printf("Write Face to: %s\n", filename);
      CVI_IVE_WriteImage(ive_handle, filename, &face_cpt_info->data[j].face_image);
    }
#endif

    face_cpt_info->data[j]._timestamp = face_cpt_info->_time;
    face_cpt_info->data[j]._capture = false;
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], image_size);
  }
  CVI_SYS_FreeI(ive_handle, &ive_frame);
  return CVIAI_SUCCESS;
}

void set_skipFQsignal(face_capture_t *face_cpt_info, cvai_face_t *face_meta, bool *skip) {
  memset(skip, 0, sizeof(bool) * face_meta->size);
  for (uint32_t i = 0; i < face_meta->size; i++) {
    if (ABS(face_meta->info[i].head_pose.yaw) > face_cpt_info->_thr_yaw ||
        ABS(face_meta->info[i].head_pose.pitch) > face_cpt_info->_thr_pitch ||
        ABS(face_meta->info[i].head_pose.roll) > face_cpt_info->_thr_roll) {
      skip[i] = true;
    }
  }
}

bool is_qualified(face_capture_t *face_cpt_info, cvai_face_info_t *face_info,
                  float current_quality) {
  if (face_info->face_quality >= face_cpt_info->_thr_quality &&
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

CVI_S32 get_ive_image_type(PIXEL_FORMAT_E enPixelFormat, IVE_IMAGE_TYPE_E *enType) {
  printf("enPixelFormat = %d\n", enPixelFormat);
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
