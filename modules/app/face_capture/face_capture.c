#include "face_capture.h"
#include "cviai_log.hpp"

#define DEFAULT_SIZE 10
#define QUALITY_THRESHOLD 0.95
#define MISS_TIME_LIMIT 100

// TODO: check correctness
void feature_copy(cvai_feature_t *src_feature, cvai_feature_t *dst_feature) {
  dst_feature->size = src_feature->size;
  dst_feature->type = src_feature->type;
  size_t type_size = getFeatureTypeSize(dst_feature->type);
  dst_feature->ptr = (int8_t *)malloc(dst_feature->size * type_size);
  memcpy(dst_feature->ptr, src_feature->ptr, dst_feature->size * type_size);
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
        if (face_cpt_info->data[j].state == MISS) {
          printf("[APP::FaceCapture] Create Face Info[%u]\n", j);
          face_cpt_info->data[j].miss_counter = 0;
          face_cpt_info->data[j].state = ALIVE;
          memcpy(&face_cpt_info->data[j].info, &face_meta->info[i], sizeof(cvai_face_info_t));
          /* set useless heap data structure to 0 */
          memset(&face_cpt_info->data[j].info.pts, 0, sizeof(cvai_pts_t));
          memset(&face_cpt_info->data[j].info.feature, 0, sizeof(cvai_feature_t));
#if 0
          if (face_meta->info[i].face_quality >= QUALITY_THRESHOLD) {
            feature_copy(&face_cpt_info->data[j].info.feature, &face_meta->info[i].feature);
            CVI_S32 ret =
                CVI_AI_GetAlignedFace(ai_handle, frame, &face_cpt_info->data[j].info.face, &face_meta->info[i]);
            if (ret != CVIAI_SUCCESS) {
              printf("AI get aligned face failed(1).\n");
              return false;
            }
          }
#endif
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
      /* if found, check whether the quality(or feature) need to be update. */
      face_cpt_info->data[match_idx].miss_counter = 0;
      if (face_meta->info[i].face_quality >= QUALITY_THRESHOLD &&
          face_meta->info[i].face_quality > face_cpt_info->data[match_idx].info.face_quality) {
        printf("[APP::FaceCapture] Update Face Info[%u]\n", match_idx);
        memcpy(&face_cpt_info->data[match_idx].info, &face_meta->info[i], sizeof(cvai_face_info_t));
        /* set useless heap data structure to 0 */
        memset(&face_cpt_info->data[match_idx].info.pts, 0, sizeof(cvai_pts_t));
        memset(&face_cpt_info->data[match_idx].info.feature, 0, sizeof(cvai_feature_t));
#if 0
        feature_copy(&face_cpt_info->data[match_idx].info.feature, &face_meta->info[i].feature);
        CVI_S32 ret = CVI_AI_GetAlignedFace(ai_handle, frame, &fq_trackers[match_idx].face,
                                            &face_meta->info[i]);
        if (ret != CVIAI_SUCCESS) {
          printf("AI get aligned face failed(2).\n");
          return false;
        }
#endif
      }
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 clean_data(face_capture_t *face_cpt_info) {
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    if (face_cpt_info->data[j].state == ALIVE &&
        face_cpt_info->data[j].miss_counter > MISS_TIME_LIMIT) {
      printf("[APP::FaceCapture] Clean Face Info[%u]\n", j);
      // free(feature);
      // free(face);
      memset(&face_cpt_info->data[j], 0, sizeof(face_cpt_data_t));
    }
  }
  return CVIAI_SUCCESS;
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

// TODO
CVI_S32 _FaceCapture_Free(face_capture_t *face_cpt_info) {
  // clean heap data
  for (uint32_t j = 0; j < face_cpt_info->size; j++) {
    printf("[APP::FaceCapture] Free Face Info[%u]\n", j);
    // if (face_cpt_info->data[j].state == ALIVE && face_cpt_info->data[j].miss_counter >
    // MISS_TIME_LIMIT) { free(feature); free(face);
    // }
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

  *face_cpt_info = new_face_cpt_info;
  return CVIAI_SUCCESS;
}

CVI_S32 _FaceCapture_Run(face_capture_t *face_cpt_info, const cviai_handle_t ai_handle,
                         VIDEO_FRAME_INFO_S *frame) {
  if (face_cpt_info == NULL) {
    LOGE("[APP::FaceCapture] is not initialized.\n");
    return CVIAI_FAILURE;
  }
  CVI_AI_Free(&face_cpt_info->last_faces);
  // cvai_face_t face_meta;
  // memset(&face_meta, 0, sizeof(cvai_face_t));
  cvai_tracker_t tracker_meta;
  memset(&tracker_meta, 0, sizeof(cvai_tracker_t));

  CVI_AI_RetinaFace(ai_handle, frame, &face_cpt_info->last_faces);
  printf("Found %x faces.\n", face_cpt_info->last_faces.size);
  CVI_AI_FaceRecognition(ai_handle, frame, &face_cpt_info->last_faces);
  CVI_AI_FaceQuality(ai_handle, frame, &face_cpt_info->last_faces);

  bool use_DeepSORT = false;
  CVI_AI_DeepSORT_Face(ai_handle, &face_cpt_info->last_faces, &tracker_meta, use_DeepSORT);
  update_data(face_cpt_info, &face_cpt_info->last_faces, &tracker_meta);
  clean_data(face_cpt_info);

  CVI_AI_Free(&tracker_meta);
  return CVIAI_SUCCESS;
}