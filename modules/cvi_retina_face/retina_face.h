#ifndef _RETINA_FACE_H_
#define _RETINA_FACE_H_

#include "cv183x_facelib_v0.0.1.h"


#ifdef __cplusplus
extern "C" {
#endif

void init_network_retina(char *model_path);
void clean_network_retina();
void retina_face_inference(VIDEO_FRAME_INFO_S *vaddr, cvi_face_t *meta, int *face_count);

#ifdef __cplusplus
}
#endif
#endif
