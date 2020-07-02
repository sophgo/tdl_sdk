#ifndef _RUN_YOLO_H_
#define _RUN_YOLO_H_

#include "cvi_object_types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void set_model_size(int model_size);
int init_network_yolov3(char *model_path);
void free_cnn_env();

void yolov3_inference(VIDEO_FRAME_INFO_S *frame, cvi_object_meta_t *meta, int det_type);

#ifdef __cplusplus
}
#endif
#endif
