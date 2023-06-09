#ifndef AI_TYPE_H_
#define AI_TYPE_H_

typedef enum {
  CVI_AI_FACE,
  CVI_AI_OBJECT,
  CVI_AI_PET,
  CVI_AI_HAND,
  CVI_AI_PERSON_VEHICLE,
  CVI_AI_MEET,
  CVI_AI_MAX,
} SAMPLE_AI_TYPE;

SAMPLE_AI_TYPE ai_param_get(void);
void ai_param_set(SAMPLE_AI_TYPE ai_type);
#endif