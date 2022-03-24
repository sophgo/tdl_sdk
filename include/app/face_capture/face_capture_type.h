#ifndef _CVIAI_APP_FACE_CAPTURE_TYPE_H_
#define _CVIAI_APP_FACE_CAPTURE_TYPE_H_

#include "core/cviai_core.h"

typedef enum { IDLE = 0, ALIVE, MISS } tracker_state_e;

typedef enum { AUTO = 0, FAST, CYCLE } capture_mode_e;

typedef enum { AREA_RATIO = 0, EYES_DISTANCE } quality_assessment_e;

typedef struct {
  cvai_face_info_t info;
  tracker_state_e state;
  uint32_t miss_counter;
  cvai_image_t image;

  bool _capture;
  uint64_t _timestamp;
  uint32_t _out_counter;
} face_cpt_data_t;

typedef struct {
  int thr_size_min;
  int thr_size_max;
  int qa_method;
  float thr_quality;
  float thr_quality_high;
  float thr_yaw;
  float thr_pitch;
  float thr_roll;

  uint32_t miss_time_limit;
  uint32_t fast_m_interval;
  uint32_t fast_m_capture_num;
  uint32_t cycle_m_interval;
  uint32_t auto_m_time_limit;
  bool auto_m_fast_cap;

  bool do_FR;
  bool capture_aligned_face;
  bool store_RGB888;
} face_capture_config_t;

typedef struct {
  capture_mode_e mode;
  bool use_fqnet;
  face_capture_config_t cfg;

  uint32_t size;
  face_cpt_data_t *data;
  cvai_face_t last_faces;
  cvai_tracker_t last_trackers;

  bool *_output;   // output signal (# = .size)
  uint64_t _time;  // timer
  uint32_t _m_limit;
} face_capture_t;

#endif  // End of _CVIAI_APP_FACE_CAPTURE_TYPE_H_
