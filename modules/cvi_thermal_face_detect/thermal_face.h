#ifndef _THERMAL_FACE_H_
#define _THERMAL_FACE_H_

#include "cv183x_facelib_v0.0.1.h"


#ifdef __cplusplus
extern "C" {
#endif

void init_network_thermal(char *model_path);
void clean_network_thermal();
void thermal_face_inference(VIDEO_FRAME_INFO_S *vaddr, cvi_face_t *meta, int *face_count);

#ifdef __cplusplus
}
#endif
#endif
