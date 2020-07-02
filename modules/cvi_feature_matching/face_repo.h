#ifndef _FACE_REPO_H_
#define _FACE_REPO_H_

#include "cv183x_facelib_v0.0.1.h"

void face_register(cvi_face_info_t *face_info, cvi_face_id_t *id);
void face_match(cvi_face_info_t *face_info,int *matched, cvi_face_id_t *id, float *score);


#endif
