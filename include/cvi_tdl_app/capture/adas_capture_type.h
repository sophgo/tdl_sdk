#ifndef _CVI_TDL_APP_ADAS_CAPTURE_TYPE_HPP_
#define _CVI_TDL_APP_ADAS_CAPTURE_TYPE_HPP_

// #ifdef __cplusplus
// extern "C" {
// #endif

#include "capture_type.h"
#include "core/cvi_tdl_core.h"

// #ifdef __cplusplus
// }

// #include <queue>

typedef struct {
  cvtdl_object_info_t info;
  tracker_state_e t_state;
  float speed[3];
  float dis;
  float dis_tmp;

  // std::queue<float> acc; //acceleration

  // float acc[3];
  int speed_counter;

  uint32_t miss_counter;

  uint32_t counter;

} adas_data_t;

typedef struct {
  uint32_t size;
  uint32_t FPS;

  adas_data_t *data;
  cvtdl_lane_t lane_meta;

  cvtdl_object_t last_objects;
  cvtdl_tracker_t last_trackers;

  uint32_t miss_time_limit;

  bool is_shifting[3];
  adas_state_e lane_state;

  float AVG_CAR_HEIGHT;   // 1.4
  float AVG_BUS_HEIGHT;   // 2.6
  float AVG_TRUCK_WIDTH;  // 2.0

  bool is_static;

} adas_info_t;

// #endif

#endif  // End of _CVI_TDL_APP_ADAS_CAPTURE_TYPE_HPP_
