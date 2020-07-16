#ifndef _CVI_FACE_HELPER_H_
#define _CVI_FACE_HELPER_H_
#include "cvai_face_types.h"

inline const char* getEmotionString(cvai_face_emotion_e emotion) {
  switch (emotion) {
    case EMOTION_HAPPY:
      return "Happy";
    case EMOTION_SURPRISE:
      return "Surprise";
    case EMOTION_FEAR:
      return "Fear";
    case EMOTION_DISGUST:
      return "Disgust";
    case EMOTION_SAD:
      return "Sad";
    case EMOTION_ANGER:
      return "Anger";
    case EMOTION_NEUTRAL:
      return "Neutral";
    default:
      return "Unknown";
  }
  return "";
}

inline const char* getGenderString(cvai_face_gender_e gender) {
  switch (gender) {
    case GENDER_MALE:
      return "Male";
    case GENDER_FEMALE:
      return "Female";
    default:
      return "Unknown";
  }
  return "";
}

inline const char* getRaceString(cvai_face_race_e race) {
  switch (race) {
    case RACE_CAUCASIAN:
      return "Caucasian";
    case RACE_BLACK:
      return "Black";
    case RACE_ASIAN:
      return "Asian";
    default:
      return "Unknown";
  }
  return "";
}

#endif