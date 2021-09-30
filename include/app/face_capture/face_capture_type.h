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
  int miss_counter;
  IVE_IMAGE_S face_image;

  bool _capture;
  uint64_t _timestamp;
} face_cpt_data_t;

typedef struct {
  capture_mode_e mode;
  uint32_t size;
  face_cpt_data_t *data;
  cvai_face_t last_faces;
  cvai_tracker_t last_trackers;
  bool *last_capture;

  float _thr_quality;
  float _thr_yaw;
  float _thr_pitch;
  float _thr_roll;

  uint64_t _time;
} face_capture_t;

#endif  // End of _CVIAI_APP_FACE_CAPTURE_TYPE_H_