#ifndef _CVIAI_APP_FACE_CAPTURE_TYPE_H_
#define _CVIAI_APP_FACE_CAPTURE_TYPE_H_

#include "core/cviai_core.h"
// #include "core/cvai_core_types.h"
// #include "face/cvai_face_types.h"
#include "ive/ive.h"

typedef enum { IDLE = 0, ALIVE, MISS } tracker_state_e;

typedef enum { AUTO = 0, FAST, CYCLE } capture_mode_e;

typedef struct {
  cvai_face_info_t info;
  tracker_state_e state;
  uint32_t miss_counter;
  IVE_IMAGE_S face_image;

  bool _capture;
  uint64_t _timestamp;
  uint32_t _out_counter;
} face_cpt_data_t;

typedef struct {
  float thr_quality;
  float thr_quality_high;
  float thr_yaw;
  float thr_pitch;
  float thr_roll;

  uint32_t miss_time_limit;
  uint32_t fast_m_interval;
  uint32_t fast_m_capture_num;
  uint32_t cycle_m_interval;
} face_capture_config_t;

typedef struct {
  capture_mode_e mode;
  uint32_t size;
  face_cpt_data_t *data;
  cvai_face_t last_faces;
  cvai_tracker_t last_trackers;
  bool *last_capture;

  face_capture_config_t cfg;

  bool write_face_stable;  // for TEST
  bool write_face_init;    // for TEST
  uint64_t _time;
} face_capture_t;

#endif  // End of _CVIAI_APP_FACE_CAPTURE_TYPE_H_