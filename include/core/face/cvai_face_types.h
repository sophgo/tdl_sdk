#ifndef _CVI_FACE_TYPES_H_
#define _CVI_FACE_TYPES_H_
#include "core/core/cvai_core_types.h"

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
} cvai_face_emotion_e;

typedef enum { GENDER_UNKNOWN = 0, GENDER_MALE, GENDER_FEMALE, GENDER_END } cvai_face_gender_e;

typedef enum {
  RACE_UNKNOWN = 0,
  RACE_CAUCASIAN,
  RACE_BLACK,
  RACE_ASIAN,
  RACE_END
} cvai_face_race_e;

typedef enum { LIVENESS_IR_LEFT = 0, LIVENESS_IR_RIGHT } cvai_liveness_ir_position_e;

typedef uint32_t cvai_face_id_t;

typedef struct {
  float quality;
  float roll;
  float pitch;
  float yaw;
} cvai_face_quality_t;

typedef struct {
  char name[128];
  cvai_bbox_t bbox;
  cvai_pts_t face_pts;
  cvai_feature_t face_feature;
  cvai_face_emotion_e emotion;
  cvai_face_gender_e gender;
  cvai_face_race_e race;
  float age;
  float liveness_score;
  float mask_score;
  cvai_face_quality_t face_quality;
} cvai_face_info_t;

typedef struct {
  int size;
  int width;
  int height;
  cvai_face_info_t* face_info;
} cvai_face_t;

#endif
