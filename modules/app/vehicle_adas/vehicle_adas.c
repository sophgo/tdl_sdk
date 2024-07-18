#include "vehicle_adas.h"

#include <math.h>
#include <sys/time.h>
#include "core/cvi_tdl_utils.h"
#include "cvi_tdl_log.hpp"
#include "cvi_venc.h"
#include "service/cvi_tdl_service.h"

CVI_S32 _ADAS_Init(adas_info_t **adas_info, uint32_t buffer_size) {
  if (*adas_info != NULL) {
    LOGW("[APP::ADAS] already exist.\n");
    return CVI_TDL_SUCCESS;
  }
  LOGI("[APP::ADAS] Initialize (Buffer Size: %u)\n", buffer_size);
  adas_info_t *new_adas_info = (adas_info_t *)malloc(sizeof(adas_info_t));
  memset(new_adas_info, 0, sizeof(adas_info_t));
  new_adas_info->size = buffer_size;
  new_adas_info->miss_time_limit = 10;
  new_adas_info->is_static = true;

  new_adas_info->data = (adas_data_t *)malloc(sizeof(adas_data_t) * buffer_size);
  memset(new_adas_info->data, 0, sizeof(adas_data_t) * buffer_size);

  *adas_info = new_adas_info;

  return CVI_SUCCESS;
}

CVI_S32 _ADAS_Free(adas_info_t *adas_info) {
  LOGI("[APP::ADAS] Free FaceCapture Data\n");
  if (adas_info != NULL) {
    /* Release tracking data */
    for (uint32_t j = 0; j < adas_info->size; j++) {
      if (adas_info->data[j].t_state != IDLE) {
        LOGI("[APP::ADAS] Clean Face Info[%u]\n", j);
        CVI_TDL_Free(&adas_info->data[j].info);
        adas_info->data[j].t_state = IDLE;
      }
    }

    free(adas_info->data);
    CVI_TDL_Free(&adas_info->last_objects);
    CVI_TDL_Free(&adas_info->last_trackers);
    CVI_TDL_Free(&adas_info->lane_meta);
    free(adas_info);
  }

  return CVI_TDL_SUCCESS;
}

float sum_acc(float *data, int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += data[i];
  }
  return sum;
}

void update_dis(cvtdl_object_info_t *info, adas_data_t *data, bool first_time) {
  float cur_dis = 582.488 * 0.5 * 1.4 / (info->bbox.y2 - info->bbox.y1);

  if (info->bbox.y2 > 900) {
    cur_dis *= 0.6;
  }

  if (!first_time) {
    cur_dis = 0.9 * data->dis + 0.1 * cur_dis;
  } else {
    data->dis_tmp = cur_dis;
  }

  data->dis = cur_dis;
  info->adas_properity.dis = cur_dis;
}

void update_apeed(cvtdl_object_info_t *info, adas_data_t *data) {
  info->adas_properity.speed = data->speed[2];

  if (data->counter >= 5) {
    float cur_speed = (data->dis - data->dis_tmp) / ((float)(data->counter - 1) / 30);

    // printf("cur_speed;%.2f,  data->speed;%.2f,   data->dis;%.2f,  data->dis_tmp;%.2f,  \n",
    // cur_speed, data->speed, data->dis, data->dis_tmp );

    if (data->speed_counter < 3) {
      data->speed_counter += 1;
    }

    data->speed[0] = data->speed[1];
    data->speed[1] = data->speed[2];
    data->speed[2] = cur_speed;

    // printf(" data->dis  %.2f   data->dis_tmp   %.2f  cur_speed: %.2f\n", data->dis,
    // data->dis_tmp, cur_speed); data->speed = cur_speed;
    info->adas_properity.speed = cur_speed;

    data->dis_tmp = data->dis;
    data->counter = 0;
  }
}

void update_state(cvtdl_object_info_t *info, adas_data_t *data, uint32_t width, bool self_static) {
  if (data->speed_counter == 3) {
    bool center = info->bbox.x1 > (float)width * 0.25 && info->bbox.x2 < (float)width * 0.75;

    float avg_speed = sum_acc(data->speed, 3) / 3.0f;
    // printf("avg_speed: %.2f\n", avg_speed);

    if (self_static && center && info->adas_properity.dis < 6.0f && avg_speed > 0.5f) {
      info->adas_properity.state = START;
    }

    if (info->adas_properity.dis < 6.0f && avg_speed < -1.0f) {
      // if (center && info->adas_properity.dis < 6.0f && avg_speed < -1.0f) {
      info->adas_properity.state = COLLISION_WARNING;
    }
  }
}

void obj_filter(cvtdl_object_t *obj_meta) {
  int size = obj_meta->size;
  if (size > 0) {
    int valid[size];
    int count = 0;
    memset(valid, 0, sizeof(valid));
    for (int i = 0; i < size; i++) {
      if ((obj_meta->info[i].bbox.x2 - obj_meta->info[i].bbox.x1) /
              (obj_meta->info[i].bbox.y2 - obj_meta->info[i].bbox.y1) <
          3) {
        valid[i] = 1;
        count += 1;
      }
    }

    if (count < size) {
      obj_meta->size = count;
      cvtdl_object_info_t *new_info = NULL;

      if (count > 0 && obj_meta->info) {
        new_info = (cvtdl_object_info_t *)malloc(sizeof(cvtdl_object_info_t) * count);
        memset(new_info, 0, sizeof(cvtdl_object_info_t) * count);

        count = 0;
        for (int oid = 0; oid < size; oid++) {
          if (valid[oid]) {
            CVI_TDL_CopyObjectInfo(&obj_meta->info[oid], &new_info[count]);
            count++;
          }
          CVI_TDL_Free(&obj_meta->info[oid]);
        }
      }

      free(obj_meta->info);
      obj_meta->info = new_info;
    }
  }
}

static CVI_S32 clean_data(adas_info_t *adas_info) {
  for (uint32_t j = 0; j < adas_info->size; j++) {
    if (adas_info->data[j].t_state == MISS) {
      LOGI("[APP::VehicleAdas] Clean Vehicle Info[%u]\n", j);
      CVI_TDL_Free(&adas_info->data[j].info);
      adas_info->data[j].info.unique_id = 0;

      adas_info->data[j].t_state = IDLE;
      adas_info->data[j].counter = 0;
      adas_info->data[j].miss_counter = 0;
      adas_info->data[j].speed_counter = 0;

      // adas_info->data[j].speed = 0;
      adas_info->data[j].dis = 0;
      adas_info->data[j].dis_tmp = 0;
    }
  }
  return CVI_TDL_SUCCESS;
}

static CVI_S32 update_lane_state(adas_info_t *adas_info, uint32_t height, uint32_t width) {
  cvtdl_lane_t *lane_meta = &adas_info->lane_meta;
  adas_info->is_shifting[0] = adas_info->is_shifting[1];
  adas_info->is_shifting[1] = adas_info->is_shifting[2];

  for (uint32_t i = 0; i < lane_meta->size; i++) {
    cvtdl_lane_point_t *point = &lane_meta->lane[i];

    float x_i = ((float)height * 0.78 - point->y[0]) / (point->y[1] - point->y[0]) *
                    (point->x[1] - point->x[0]) +
                point->x[0];

    if (x_i > (float)width * 0.4 && x_i < (float)width * 0.6) {
      adas_info->is_shifting[2] = true;
      break;
    }
    adas_info->is_shifting[2] = false;
  }

  if (adas_info->is_shifting[0] && adas_info->is_shifting[1] && adas_info->is_shifting[2]) {
    adas_info->lane_state = 1;
  } else {
    adas_info->lane_state = 0;
  }

  // printf("adas_info->lane_state: %d\n", adas_info->lane_state);
}

static CVI_S32 update_data(cvitdl_handle_t tdl_handle, adas_info_t *adas_info,
                           VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *obj_meta,
                           cvtdl_tracker_t *tracker_meta) {
  LOGI("[APP::VehicleAdas] Update Data\n");

  for (uint32_t i = 0; i < obj_meta->size; i++) {
    uint64_t trk_id = obj_meta->info[i].unique_id;
    int match_idx = -1;
    int idle_idx = -1;
    int update_idx = -1;
    //  printf("obj_meta->size: %d\n", adas_info->size);

    for (uint32_t j = 0; j < adas_info->size; j++) {
      if (adas_info->data[j].t_state == ALIVE && adas_info->data[j].info.unique_id == trk_id) {
        match_idx = (int)j;
        break;
      }
    }

    if (match_idx != -1) {
      adas_info->data[match_idx].miss_counter = 0;
      update_dis(&obj_meta->info[i], &adas_info->data[match_idx], false);

      update_idx = match_idx;
    } else {
      for (uint32_t j = 0; j < adas_info->size; j++) {
        if (adas_info->data[j].t_state == IDLE) {
          idle_idx = (int)j;
          update_dis(&obj_meta->info[i], &adas_info->data[idle_idx], true);
          update_idx = idle_idx;
          break;
        }
      }
    }

    if (match_idx == -1 && idle_idx == -1) {
      LOGD("no valid buffer\n");
      continue;
    }

    memcpy(&adas_info->data[update_idx].info, &obj_meta->info[i], sizeof(cvtdl_object_info_t));
    adas_info->data[update_idx].t_state = ALIVE;

    adas_info->data[update_idx].counter += 1;

    if (match_idx != -1) {
      update_apeed(&obj_meta->info[i], &adas_info->data[match_idx]);
      update_state(&obj_meta->info[i], &adas_info->data[match_idx], obj_meta->width, true);
    }
  }

  for (uint32_t j = 0; j < adas_info->size; j++) {
    bool found = false;
    for (uint32_t k = 0; k < tracker_meta->size; k++) {
      if (adas_info->data[j].info.unique_id == tracker_meta->info[k].id) {
        found = true;
        break;
      }
    }

    if (!found && adas_info->data[j].info.unique_id != 0) {
      adas_info->data[j].miss_counter += 1;
      adas_info->data[j].counter += 1;

      if (adas_info->data[j].miss_counter >= adas_info->miss_time_limit) {
        LOGD("to delete track:%u\n", (uint32_t)adas_info->data[j].info.unique_id);
        adas_info->data[j].t_state = MISS;
      }
    }
  }

  return CVI_TDL_SUCCESS;
}

CVI_S32 _ADAS_Run(adas_info_t *adas_info, const cvitdl_handle_t tdl_handle,
                  VIDEO_FRAME_INFO_S *frame) {
  if (adas_info == NULL) {
    LOGE("[APP::VehicleAdas] is not initialized.\n");
    return CVI_TDL_FAILURE;
  }

  if (frame->stVFrame.u32Length[0] == 0) {
    LOGE("[APP::VehicleAdas] got empty frame.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret;
  ret = clean_data(adas_info);
  if (ret != CVI_TDL_SUCCESS) {
    LOGE("[APP::VehicleAdas] clean data failed.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_TDL_Free(&adas_info->last_trackers);
  CVI_TDL_Free(&adas_info->last_objects);

  if (frame->stVFrame.u32Length[0] == 0) {
    LOGE("input frame turn into empty\n");
    return CVI_TDL_FAILURE;
  }

  if (CVI_SUCCESS != CVI_TDL_PersonVehicle_Detection(tdl_handle, frame, &adas_info->last_objects)) {
    // CVI_TDL_Release_VideoFrame(tdl_handle, CVI_TDL_SUPPORTED_MODEL_PERSON_VEHICLE_DETECTION,
    // frame,
    //  true);
    printf("PersonVehicle detection failed\n");
    return CVI_TDL_FAILURE;
  }

  obj_filter(&adas_info->last_objects);

  CVI_TDL_DeepSORT_Obj(tdl_handle, &adas_info->last_objects, &adas_info->last_trackers, false);

  if (CVI_SUCCESS != CVI_TDL_Lane_Det(tdl_handle, frame, &adas_info->lane_meta)) {
    // CVI_TDL_Release_VideoFrame(tdl_handle, CVI_TDL_SUPPORTED_MODEL_LANE_DET, frame, true);
    printf("lane detection failed\n");
    return CVI_TDL_FAILURE;
  }

  update_lane_state(adas_info, frame->stVFrame.u32Height, frame->stVFrame.u32Width);

  ret = update_data(tdl_handle, adas_info, frame, &adas_info->last_objects,
                    &adas_info->last_trackers);
  if (ret != CVI_TDL_SUCCESS) {
    LOGE("[APP::ADAS] update face failed.\n");
    return CVI_TDL_FAILURE;
  }

  return CVI_TDL_SUCCESS;
}