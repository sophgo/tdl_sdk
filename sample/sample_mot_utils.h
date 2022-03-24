#ifndef SAMPLE_MOT_UTILS_H
#define SAMPLE_MOT_UTILS_H

#include "cviai.h"

typedef enum { FACE = 0, PERSON, VEHICLE, PET } TARGET_TYPE_e;

CVI_U32 GET_TARGET_TYPE(TARGET_TYPE_e *target_type, char *text);
CVI_U32 GET_PREDEFINED_CONFIG(cvai_deepsort_config_t *ds_conf, TARGET_TYPE_e target_type);

#endif