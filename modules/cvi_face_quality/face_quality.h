#ifndef _FACE_QUALITY_H_
#define _FACE_QUALITY_H_

#include "cv183x_facelib_v0.0.1.h"


#ifdef __cplusplus
extern "C" {
#endif

void init_network_face_quality(char *model_path);
void clean_network_face_quality();
void face_quality_inference(VIDEO_FRAME_INFO_S *frame, cvi_face_t *meta);

#ifdef __cplusplus
}
#endif
#endif
