#ifndef _LIVENESS_H_
#define _LIVENESS_H

#include "cv183x_facelib_v0.0.1.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LIVENESS_THRESHOLD      (128 / 1.00000488758) / 255.0
#define CROP_NUM                9
//#define DEFAULT_LIVENESS_MODEL_PATH     "/mnt/data/bmodel/dual_liveness.bmodel"
int init_network_liveness(char *model_path);
void clean_network_liveness();
void liveness_inference(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *sink_buffer, cvi_face_t *meta);

#ifdef __cplusplus
}
#endif
#endif
