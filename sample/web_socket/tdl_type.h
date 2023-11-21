#ifndef AI_TYPE_H_
#define AI_TYPE_H_

typedef enum {
  CVI_TDL_FACE,
  CVI_TDL_OBJECT,
  CVI_TDL_PET,
  CVI_TDL_HAND,
  CVI_TDL_PERSON_VEHICLE,
  CVI_TDL_MEET,
  CVI_TDL_MAX,
} SAMPLE_TDL_TYPE;

SAMPLE_TDL_TYPE ai_param_get(void);
void ai_param_set(SAMPLE_TDL_TYPE ai_type);
#endif