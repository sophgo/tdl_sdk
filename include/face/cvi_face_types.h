#ifndef _CVI_FACE_TYPES_H_
#define _CVI_FACE_TYPES_H_
#include "core/cvi_core_types.h"

typedef enum {
  EMOTION_UNKNOWN = 0,
  EMOTION_HAPPY,
  EMOTION_SURPRISE,
  EMOTION_FEAR,
  EMOTION_DISGUST,
  EMOTION_SAD,
  EMOTION_ANGER,
  EMOTION_NEUTRAL,
  EMOTION_END
} cvi_face_emotion_e;

typedef enum { GENDER_UNKNOWN = 0, GENDER_MALE, GENDER_FEMALE, GENDER_END } cvi_face_gender_e;

typedef enum { RACE_UNKNOWN = 0, RACE_CAUCASIAN, RACE_BLACK, RACE_ASIAN, RACE_END } cvi_face_race_e;

typedef uint32_t cvi_face_id_t;

typedef struct {
  char name[128];
  cvi_detect_rect_t bbox;
  cvi_pts_t face_pts;
  cvi_feature_t face_feature;
  cvi_face_emotion_e emotion;
  cvi_face_gender_e gender;
  cvi_face_race_e race;
  float age;
  float liveness_score;
  float mask_score;
} cvi_face_info_t;

typedef struct {
  int size;
  int width;
  int height;
  cvi_face_info_t* face_info;
} cvi_face_t;

#endif