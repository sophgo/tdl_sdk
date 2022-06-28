#include "person_capture.h"
#include "default_config.h"

#include <math.h>
#include "core/cviai_utils.h"
#include "cviai_log.hpp"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MEMORY_LIMIT (16 * 1024 * 1024) /* example: 16MB */

#define CAPTURE_TARGET_LIVE_TIME_EXTEND 5
#define UPDATE_VALUE_MIN 0.05
// TODO: Use cooldown to avoid too much updating
// #define UPDATE_COOLDOWN 3

/* person capture functions (core) */
static CVI_S32 update_data(person_capture_t *person_cpt_info, cvai_object_t *obj_meta,
                           cvai_tracker_t *tracker_meta, float *quality);
static CVI_S32 clean_data(person_capture_t *person_cpt_info);
static CVI_S32 capture_target(person_capture_t *person_cpt_info, VIDEO_FRAME_INFO_S *frame,
                              cvai_object_t *obj_meta);
static void quality_assessment(person_capture_t *person_cpt_info, float *quality);

/* person capture functions (helper) */
static bool is_qualified(person_capture_t *person_cpt_info, float quality, float current_quality);

/* other helper functions */
static bool IS_MEMORY_ENOUGH(uint32_t mem_limit, uint64_t mem_used, cvai_image_t *current_image,
                             cvai_bbox_t *new_bbox, PIXEL_FORMAT_E fmt);
static void SUMMARY(person_capture_t *person_cpt_info, uint64_t *size, bool show_detail);
static void SHOW_CONFIG(person_capture_config_t *cfg);

CVI_S32 _PersonCapture_Free(person_capture_t *person_cpt_info) {
  LOGI("[APP::PersonCapture] Free PersonCapture Data\n");
  if (person_cpt_info != NULL) {
    _PersonCapture_CleanAll(person_cpt_info);

    free(person_cpt_info->data);
    CVI_AI_Free(&person_cpt_info->last_objects);
    CVI_AI_Free(&person_cpt_info->last_trackers);
    if (person_cpt_info->last_quality != NULL) {
      free(person_cpt_info->last_quality);
      person_cpt_info->last_quality = NULL;
    }
    free(person_cpt_info->_output);

    free(person_cpt_info);
  }
  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_Init(person_capture_t **person_cpt_info, uint32_t buffer_size) {
  if (*person_cpt_info != NULL) {
    LOGW("[APP::PersonCapture] already exist.\n");
    return CVIAI_SUCCESS;
  }
  LOGI("[APP::PersonCapture] Initialize (Buffer Size: %u)\n", buffer_size);
  person_capture_t *new_person_cpt_info = (person_capture_t *)malloc(sizeof(person_capture_t));
  memset(new_person_cpt_info, 0, sizeof(person_capture_t));
  new_person_cpt_info->last_quality = NULL;
  new_person_cpt_info->size = buffer_size;
  new_person_cpt_info->data = (person_cpt_data_t *)malloc(sizeof(person_cpt_data_t) * buffer_size);
  memset(new_person_cpt_info->data, 0, sizeof(person_cpt_data_t) * buffer_size);

  new_person_cpt_info->_output = (bool *)malloc(sizeof(bool) * buffer_size);
  memset(new_person_cpt_info->_output, 0, sizeof(bool) * buffer_size);

  _PersonCapture_GetDefaultConfig(&new_person_cpt_info->cfg);
  new_person_cpt_info->_m_limit = MEMORY_LIMIT;

  *person_cpt_info = new_person_cpt_info;
  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_QuickSetUp(cviai_handle_t ai_handle, person_capture_t *person_cpt_info,
                                  const char *od_model_name, const char *od_model_path,
                                  const char *reid_model_path) {
  CVI_S32 ret = CVIAI_SUCCESS;

  if (strcmp(od_model_name, "mobiledetv2-person-vehicle") == 0) {
    person_cpt_info->od_model_index = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE;
  } else if (strcmp(od_model_name, "mobiledetv2-person-pets") == 0) {
    person_cpt_info->od_model_index = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS;
  } else if (strcmp(od_model_name, "mobiledetv2-coco80") == 0) {
    person_cpt_info->od_model_index = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80;
  } else if (strcmp(od_model_name, "mobiledetv2-pedestrian") == 0) {
    person_cpt_info->od_model_index = CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN;
  } else if (strcmp(od_model_name, "yolov3") == 0) {
    person_cpt_info->od_model_index = CVI_AI_SUPPORTED_MODEL_YOLOV3;
  } else {
    return CVIAI_FAILURE;
  }

  ret |= CVI_AI_OpenModel(ai_handle, person_cpt_info->od_model_index, od_model_path);
  CVI_AI_SetSkipVpssPreprocess(ai_handle, person_cpt_info->od_model_index, false);
  if (reid_model_path != NULL) {
    ret |= CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, reid_model_path);
    CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);
  }
  if (ret != CVIAI_SUCCESS) {
    LOGE("PersonCapture QuickSetUp failed\n");
    return ret;
  }
  person_cpt_info->enable_DeepSORT = reid_model_path != NULL;

  /* Init DeepSORT */
  CVI_AI_DeepSORT_Init(ai_handle, false);

  cvai_deepsort_config_t ds_conf;
  CVI_AI_DeepSORT_GetDefaultConfig(&ds_conf);
  CVI_AI_DeepSORT_SetConfig(ai_handle, &ds_conf, -1, false);

  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_GetDefaultConfig(person_capture_config_t *cfg) {
  cfg->miss_time_limit = MISS_TIME_LIMIT;
  cfg->thr_area_base = THRESHOLD_AREA_BASE;
  cfg->thr_area_min = THRESHOLD_AREA_MIN;
  cfg->thr_area_max = THRESHOLD_AREA_MAX;
  cfg->thr_aspect_ratio_min = THRESHOLD_ASPECT_RATIO_MIN;
  cfg->thr_aspect_ratio_max = THRESHOLD_ASPECT_RATIO_MAX;
  cfg->thr_quality = QUALITY_THRESHOLD;

  cfg->fast_m_interval = FAST_MODE_INTERVAL;
  cfg->fast_m_capture_num = FAST_MODE_CAPTURE_NUM;
  cfg->cycle_m_interval = CYCLE_MODE_INTERVAL;
  cfg->auto_m_time_limit = AUTO_MODE_TIME_LIMIT;
  cfg->auto_m_fast_cap = true;

  cfg->store_RGB888 = false;

  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_SetConfig(person_capture_t *person_cpt_info, person_capture_config_t *cfg,
                                 cviai_handle_t ai_handle) {
  memcpy(&person_cpt_info->cfg, cfg, sizeof(person_capture_config_t));
  cvai_deepsort_config_t deepsort_conf;
  CVI_AI_DeepSORT_GetConfig(ai_handle, &deepsort_conf, -1);
  deepsort_conf.ktracker_conf.max_unmatched_num = cfg->miss_time_limit;
  CVI_AI_DeepSORT_SetConfig(ai_handle, &deepsort_conf, -1, false);
  SHOW_CONFIG(&person_cpt_info->cfg);
  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_Run(person_capture_t *person_cpt_info, const cviai_handle_t ai_handle,
                           VIDEO_FRAME_INFO_S *frame) {
  if (person_cpt_info == NULL) {
    LOGE("[APP::PersonCapture] is not initialized.\n");
    return CVIAI_FAILURE;
  }
  LOGI("[APP::PersonCapture] RUN (MODE: %d, ReID: %d)\n", person_cpt_info->mode,
       person_cpt_info->enable_DeepSORT);
  CVI_S32 ret;
  ret = clean_data(person_cpt_info);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::PersonCapture] clean data failed.\n");
    return CVIAI_FAILURE;
  }
  CVI_AI_Free(&person_cpt_info->last_objects);
  CVI_AI_Free(&person_cpt_info->last_trackers);
  /* set output signal to 0. */
  memset(person_cpt_info->_output, 0, sizeof(bool) * person_cpt_info->size);

  switch (person_cpt_info->od_model_index) {
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_VEHICLE:
      if (CVIAI_SUCCESS !=
          CVI_AI_MobileDetV2_Person_Vehicle(ai_handle, frame, &person_cpt_info->last_objects)) {
        return CVIAI_FAILURE;
      }
      break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PERSON_PETS:
      CVI_AI_MobileDetV2_Person_Pets(ai_handle, frame, &person_cpt_info->last_objects);
      break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_COCO80:
      CVI_AI_MobileDetV2_COCO80(ai_handle, frame, &person_cpt_info->last_objects);
      break;
    case CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN:
      CVI_AI_MobileDetV2_Pedestrian(ai_handle, frame, &person_cpt_info->last_objects);
      break;
    case CVI_AI_SUPPORTED_MODEL_YOLOV3:
      if (CVIAI_SUCCESS != CVI_AI_Yolov3(ai_handle, frame, &person_cpt_info->last_objects)) {
        return CVIAI_FAILURE;
      }
      break;
    default:
      LOGE("unknown object detection model index.");
      return CVIAI_FAILURE;
  }

  if (person_cpt_info->enable_DeepSORT) {
    if (CVIAI_SUCCESS != CVI_AI_OSNet(ai_handle, frame, &person_cpt_info->last_objects)) {
      return CVIAI_FAILURE;
    }
  }
  if (CVIAI_SUCCESS != CVI_AI_DeepSORT_Obj(ai_handle, &person_cpt_info->last_objects,
                                           &person_cpt_info->last_trackers,
                                           person_cpt_info->enable_DeepSORT)) {
    return CVIAI_FAILURE;
  }

  if (person_cpt_info->last_quality != NULL) {
    free(person_cpt_info->last_quality);
    person_cpt_info->last_quality = NULL;
  }
  person_cpt_info->last_quality =
      (float *)malloc(sizeof(float) * person_cpt_info->last_objects.size);
  memset(person_cpt_info->last_quality, 0, sizeof(float) * person_cpt_info->last_objects.size);

  quality_assessment(person_cpt_info, person_cpt_info->last_quality);
#if 0
  for (uint32_t i = 0; i < person_cpt_info->last_objects.size; i++) {
    cvai_bbox_t *bbox = &person_cpt_info->last_objects.info[i].bbox;
    printf("[%u][%d] quality[%.2f], x1[%.2f], y1[%.2f], x2[%.2f], y2[%.2f], h[%.2f], w[%.2f]\n", i,
           person_cpt_info->last_objects.info[i].classes, person_cpt_info->last_quality[i],
           bbox->x1, bbox->y1, bbox->x2, bbox->y2, (bbox->y2 - bbox->y1), (bbox->x2 - bbox->x1));
  }
#endif

  ret = update_data(person_cpt_info, &person_cpt_info->last_objects,
                    &person_cpt_info->last_trackers, person_cpt_info->last_quality);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::PersonCapture] update data failed.\n");
    return CVIAI_FAILURE;
  }
  ret = capture_target(person_cpt_info, frame, &person_cpt_info->last_objects);
  if (ret != CVIAI_SUCCESS) {
    LOGE("[APP::PersonCapture] capture target failed.\n");
    return CVIAI_FAILURE;
  }

#if 0
  uint64_t mem_used;
  SUMMARY(person_cpt_info, &mem_used, true);
#endif

  /* update timestamp*/
  person_cpt_info->_time =
      (person_cpt_info->_time == 0xffffffffffffffff) ? 0 : person_cpt_info->_time + 1;

  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_SetMode(person_capture_t *person_cpt_info, capture_mode_e mode) {
  person_cpt_info->mode = mode;
  return CVIAI_SUCCESS;
}

CVI_S32 _PersonCapture_CleanAll(person_capture_t *person_cpt_info) {
  /* Release tracking data */
  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    if (person_cpt_info->data[j].state != IDLE) {
      LOGI("[APP::PersonCapture] Clean Person Info[%u]\n", j);
      CVI_AI_Free(&person_cpt_info->data[j].image);
      CVI_AI_Free(&person_cpt_info->data[j].info);
      person_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

static void quality_assessment(person_capture_t *person_cpt_info, float *quality) {
  for (uint32_t i = 0; i < person_cpt_info->last_objects.size; i++) {
    cvai_bbox_t *bbox = &person_cpt_info->last_objects.info[i].bbox;
    float aspect_ratio = (bbox->y2 - bbox->y1) / (bbox->x2 - bbox->x1);
    if (aspect_ratio < person_cpt_info->cfg.thr_aspect_ratio_min ||
        aspect_ratio > person_cpt_info->cfg.thr_aspect_ratio_max) {
      quality[i] = 0.;
      continue;
    }
    float area = (bbox->y2 - bbox->y1) * (bbox->x2 - bbox->x1);
    if (area < person_cpt_info->cfg.thr_area_min || area > person_cpt_info->cfg.thr_area_max) {
      quality[i] = 0.;
      continue;
    }
    float area_score = MIN(1.0, area / person_cpt_info->cfg.thr_area_base);
    quality[i] = area_score;
  }
  return;
}

static CVI_S32 update_data(person_capture_t *person_cpt_info, cvai_object_t *obj_meta,
                           cvai_tracker_t *tracker_meta, float *quality) {
  LOGI("[APP::PersonCapture] Update Data\n");
  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    if (person_cpt_info->data[j].state == ALIVE) {
      person_cpt_info->data[j].miss_counter += 1;
    }
  }
  for (uint32_t i = 0; i < tracker_meta->size; i++) {
    /* we only consider the stable tracker in this sample code. */
    if (tracker_meta->info[i].state != CVI_TRACKER_STABLE) {
      continue;
    }

    uint64_t trk_id = obj_meta->info[i].unique_id;
    /* check whether the tracker id exist or not. */
    int match_idx = -1;
    for (uint32_t j = 0; j < person_cpt_info->size; j++) {
      if (person_cpt_info->data[j].state == ALIVE &&
          person_cpt_info->data[j].info.unique_id == trk_id) {
        match_idx = (int)j;
        break;
      }
    }
    if (match_idx == -1) {
      /* if not found, create new one. */
      bool is_created = false;
      /* search available index for new tracker. */
      for (uint32_t j = 0; j < person_cpt_info->size; j++) {
        if (person_cpt_info->data[j].state == IDLE && person_cpt_info->data[j]._capture == false) {
          // LOGI("[APP::PersonCapture] Create Target Info[%u]\n", j);
          person_cpt_info->data[j].miss_counter = 0;
          memcpy(&person_cpt_info->data[j].info, &obj_meta->info[i], sizeof(cvai_object_info_t));
          person_cpt_info->data[j].quality = quality[i];
          /* set useless heap data structure to 0 */
          memset(&person_cpt_info->data[j].info.feature, 0, sizeof(cvai_feature_t));

          /* always capture target in the first frame. */
          person_cpt_info->data[j]._capture = true;
          person_cpt_info->data[j]._timestamp = person_cpt_info->_time;
          person_cpt_info->data[j]._out_counter = 0;
          is_created = true;
          break;
        }
      }
      /* fail to create */
      if (!is_created) {
        LOGW("[APP::PersonCapture] Buffer overflow! (Ignore person[%u])\n", i);
      }
    } else {
      person_cpt_info->data[match_idx].miss_counter = 0;
      bool capture = false;
      uint64_t _time = person_cpt_info->_time - person_cpt_info->data[match_idx]._timestamp;
      float current_quality = person_cpt_info->data[match_idx].quality;
      switch (person_cpt_info->mode) {
        case AUTO: {
          bool time_out = (person_cpt_info->cfg.auto_m_time_limit != 0) &&
                          (_time > person_cpt_info->cfg.auto_m_time_limit);
          if (!time_out) {
            float update_quality_threshold = (current_quality < person_cpt_info->cfg.thr_quality)
                                                 ? 0.
                                                 : MIN(1.0, current_quality + UPDATE_VALUE_MIN);
            capture = is_qualified(person_cpt_info, quality[i], update_quality_threshold);
          }
        } break;
        case FAST: {
          if (person_cpt_info->data[match_idx]._out_counter <
              person_cpt_info->cfg.fast_m_capture_num) {
            if (_time < person_cpt_info->cfg.fast_m_interval) {
              float update_quality_threshold;
              if (current_quality < person_cpt_info->cfg.thr_quality) {
                update_quality_threshold = -1;
              } else {
                update_quality_threshold = ((current_quality + UPDATE_VALUE_MIN) >= 1.0)
                                               ? current_quality
                                               : current_quality + UPDATE_VALUE_MIN;
              }
              capture = is_qualified(person_cpt_info, quality[i], update_quality_threshold);
            } else {
              capture = true;
              person_cpt_info->data[match_idx]._timestamp = person_cpt_info->_time;
            }
          }
        } break;
        case CYCLE: {
          if (_time < person_cpt_info->cfg.cycle_m_interval) {
            float update_quality_threshold;
            if (current_quality < person_cpt_info->cfg.thr_quality) {
              update_quality_threshold = -1;
            } else {
              update_quality_threshold = ((current_quality + UPDATE_VALUE_MIN) >= 1.0)
                                             ? current_quality
                                             : current_quality + UPDATE_VALUE_MIN;
            }
            capture = is_qualified(person_cpt_info, quality[i], update_quality_threshold);
          } else {
            capture = true;
            person_cpt_info->data[match_idx]._timestamp = person_cpt_info->_time;
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
        LOGI("[APP::PersonCapture] Update Person Info[%u]\n", match_idx);
        memcpy(&person_cpt_info->data[match_idx].info, &obj_meta->info[i],
               sizeof(cvai_object_info_t));
        /* set useless heap data structure to 0 */
        memset(&person_cpt_info->data[match_idx].info.feature, 0, sizeof(cvai_feature_t));

        person_cpt_info->data[match_idx].quality = quality[i];
        person_cpt_info->data[match_idx]._capture = true;
      }
      switch (person_cpt_info->mode) {
        case AUTO: {
          /* We use fast interval for the first capture */
          if (person_cpt_info->cfg.auto_m_fast_cap) {
            if (_time >= person_cpt_info->cfg.fast_m_interval &&
                person_cpt_info->data[match_idx]._out_counter < 1 &&
                is_qualified(person_cpt_info, person_cpt_info->data[match_idx].quality, -1)) {
              person_cpt_info->_output[match_idx] = true;
              person_cpt_info->data[match_idx]._out_counter += 1;
            }
          }
          if (person_cpt_info->cfg.auto_m_time_limit != 0) {
            /* Time's up */
            if (_time == person_cpt_info->cfg.auto_m_time_limit &&
                is_qualified(person_cpt_info, person_cpt_info->data[match_idx].quality, -1)) {
              person_cpt_info->_output[match_idx] = true;
              person_cpt_info->data[match_idx]._out_counter += 1;
            }
          }
        } break;
        case FAST: {
          if (person_cpt_info->data[match_idx]._out_counter <
              person_cpt_info->cfg.fast_m_capture_num) {
            if (_time == person_cpt_info->cfg.fast_m_interval - 1 &&
                is_qualified(person_cpt_info, person_cpt_info->data[match_idx].quality, -1)) {
              person_cpt_info->_output[match_idx] = true;
              person_cpt_info->data[match_idx]._out_counter += 1;
            }
          }
        } break;
        case CYCLE: {
          if (_time == person_cpt_info->cfg.cycle_m_interval - 1 &&
              is_qualified(person_cpt_info, person_cpt_info->data[match_idx].quality, -1)) {
            person_cpt_info->_output[match_idx] = true;
            person_cpt_info->data[match_idx]._out_counter += 1;
          }
        } break;
        default:
          return CVIAI_FAILURE;
      }
    }
  }

  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    /* NOTE: For more flexible application, we do not remove the tracker immediately when time out
     */
    if (person_cpt_info->data[j].state == ALIVE &&
        person_cpt_info->data[j].miss_counter >
            person_cpt_info->cfg.miss_time_limit + CAPTURE_TARGET_LIVE_TIME_EXTEND) {
      person_cpt_info->data[j].state = MISS;
      switch (person_cpt_info->mode) {
        case AUTO: {
          if (is_qualified(person_cpt_info, person_cpt_info->data[j].quality, -1)) {
            person_cpt_info->_output[j] = true;
          }
        } break;
        case FAST:
        case CYCLE: {
          /* at least capture 1 target */
          if (is_qualified(person_cpt_info, person_cpt_info->data[j].quality, -1) &&
              person_cpt_info->data[j]._out_counter == 0) {
            person_cpt_info->_output[j] = true;
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

static CVI_S32 clean_data(person_capture_t *person_cpt_info) {
  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    if (person_cpt_info->data[j].state == MISS) {
      LOGI("[APP::PersonCapture] Clean Person Info[%u]\n", j);
      CVI_AI_Free(&person_cpt_info->data[j].image);
      CVI_AI_Free(&person_cpt_info->data[j].info);
      person_cpt_info->data[j].state = IDLE;
    }
  }
  return CVIAI_SUCCESS;
}

static CVI_S32 capture_target(person_capture_t *person_cpt_info, VIDEO_FRAME_INFO_S *frame,
                              cvai_object_t *obj_meta) {
  LOGI("[APP::PersonCapture] Capture Target\n");
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888_PLANAR &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE("Pixel format [%d] is not supported.\n", frame->stVFrame.enPixelFormat);
    return CVIAI_ERR_INVALID_ARGS;
  }

  bool capture = false;
  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    if (person_cpt_info->data[j]._capture) {
      capture = true;
      break;
    }
  }
  if (!capture) {
    return CVIAI_SUCCESS;
  }
  /* Estimate memory used */
  uint64_t mem_used;
  SUMMARY(person_cpt_info, &mem_used, false);

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

  for (uint32_t j = 0; j < person_cpt_info->size; j++) {
    if (!(person_cpt_info->data[j]._capture)) {
      continue;
    }
    bool first_capture = false;
    if (person_cpt_info->data[j].state != ALIVE) {
      /* first capture */
      person_cpt_info->data[j].state = ALIVE;
      first_capture = true;
    }
    LOGI("Capture Target[%u] (%s)!\n", j, (first_capture) ? "INIT" : "UPDATE");

    /* Check remaining memory space */
    if (!IS_MEMORY_ENOUGH(person_cpt_info->_m_limit, mem_used, &person_cpt_info->data[j].image,
                          &person_cpt_info->data[j].info.bbox, frame->stVFrame.enPixelFormat)) {
      LOGW("Memory is not enough. (drop)\n");
      if (first_capture) {
        person_cpt_info->data[j].state = IDLE;
      }
      continue;
    }
    CVI_AI_Free(&person_cpt_info->data[j].image);

    CVI_AI_CropImage(frame, &person_cpt_info->data[j].image, &person_cpt_info->data[j].info.bbox,
                     person_cpt_info->cfg.store_RGB888);

    person_cpt_info->data[j]._capture = false;
  }
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], image_size);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return CVIAI_SUCCESS;
}

static bool is_qualified(person_capture_t *person_cpt_info, float quality, float current_quality) {
  if (quality >= person_cpt_info->cfg.thr_quality && quality > current_quality) {
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

static void SUMMARY(person_capture_t *person_cpt_info, uint64_t *size, bool show_detail) {
  *size = 0;
  if (show_detail) {
    printf("@@@@ SUMMARY @@@@\n");
  }
  for (uint32_t i = 0; i < person_cpt_info->size; i++) {
    tracker_state_e state = person_cpt_info->data[i].state;
    if (state == IDLE) {
      if (show_detail) {
        printf("Data[%u] state[IDLE]\n", i);
      }
    } else {
      uint64_t m = person_cpt_info->data[i].image.length[0];
      m += person_cpt_info->data[i].image.length[1];
      m += person_cpt_info->data[i].image.length[2];
      if (show_detail) {
        printf("Data[%u] state[%s], h[%u], w[%u], size[%" PRIu64 "]\n", i,
               (state == ALIVE) ? "ALIVE" : "MISS", person_cpt_info->data[i].image.height,
               person_cpt_info->data[i].image.width, m);
      }
      *size += m;
    }
  }
  if (show_detail) {
    printf("MEMORY USED: %" PRIu64 "\n\n", *size);
  }
}

static void SHOW_CONFIG(person_capture_config_t *cfg) {
  printf("@@@ Person Capture Config @@@\n");
  printf(" - Miss Time Limit:   : %u\n", cfg->miss_time_limit);
  printf(" - Thr Area (Base)    : %d\n", cfg->thr_area_base);
  printf(" - Thr Area (Min)     : %d\n", cfg->thr_area_min);
  printf(" - Thr Area (Max)     : %d\n", cfg->thr_area_max);
  printf(" - Thr Aspect Ratio (Min) : %.2f\n", cfg->thr_aspect_ratio_min);
  printf(" - Thr Aspect Ratio (Max) : %.2f\n", cfg->thr_aspect_ratio_max);
  printf(" - Thr Quality        : %.2f\n", cfg->thr_quality);
  printf("[Fast] Interval     : %u\n", cfg->fast_m_interval);
  printf("[Fast] Capture Num  : %u\n", cfg->fast_m_capture_num);
  printf("[Cycle] Interval    : %u\n", cfg->cycle_m_interval);
  printf("[Auto] Time Limit   : %u\n\n", cfg->auto_m_time_limit);
  printf("[Auto] Fast Capture : %s\n\n", cfg->auto_m_fast_cap ? "True" : "False");
  printf(" - Store RGB888         : %s\n\n", cfg->store_RGB888 ? "True" : "False");
  return;
}
