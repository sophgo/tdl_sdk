#ifndef _FACE_ATTRIBUTE_H_
#define _FACE_ATTRIBUTE_H_

#include "cvi_face_types.hpp"

#define FACE_ATTRIBUTE_N                    1
#define FACE_ATTRIBUTE_C                    3
#define FACE_ATTRIBUTE_WIDTH                112
#define FACE_ATTRIBUTE_HEIGHT               112
#define FACE_ATTRIBUTE_MEAN                 (-0.99609375)
#define FACE_ATTRIBUTE_INPUT_THRESHOLD      (1/128.0)

int init_network_face_attribute(char *model_path);
void clean_network_face_attribute();
void face_attribute_inference(VIDEO_FRAME_INFO_S *frame, cvi_face_t *meta);

#endif
