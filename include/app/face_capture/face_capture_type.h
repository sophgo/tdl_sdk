#ifndef _CVIAI_APP_FACE_CAPTURE_TYPE_H_
#define _CVIAI_APP_FACE_CAPTURE_TYPE_H_

#include "core/cviai_core.h"
// #include "core/cvai_core_types.h"
// #include "face/cvai_face_types.h"

typedef enum { MISS = 0, ALIVE } tracker_state_e;

typedef enum { AUTO = 0, FAST, CYCLE } capture_mode_e;

typedef struct {
  cvai_face_info_t info;
  tracker_state_e state;
  int miss_counter;
  // VIDEO_FRAME_INFO_S frame;
} face_cpt_data_t;

typedef struct {
  capture_mode_e mode;
  uint32_t size;
  face_cpt_data_t *data;
  cvai_face_t last_faces;

  int _counter;
} face_capture_t;

#endif  // End of _CVIAI_APP_FACE_CAPTURE_TYPE_H_