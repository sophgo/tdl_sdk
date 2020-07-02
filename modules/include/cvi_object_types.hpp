#ifndef _CVI_OBJECT_TYPES_H_
#define _CVI_OBJECT_TYPES_H_

#include "cvi_face_types.hpp"

typedef struct {
    cvi_face_detect_rect_t bbox;
    char name[128];
    int classes;
} cvi_object_info_t;

typedef struct {
    int size;
    int width;
    int height;
    cvi_object_info_t *objects;
} cvi_object_meta_t;

#endif