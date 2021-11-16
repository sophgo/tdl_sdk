#pragma once
#include "cviai.h"
namespace cviai {
namespace unitest {

void init_face_meta(cvai_face_t *meta, uint32_t size);

void init_obj_meta(cvai_object_t *meta, uint32_t size, uint32_t height, uint32_t width,
                   int class_id);

void init_vehicle_meta(cvai_object_t *meta);

}  // namespace unitest
}  // namespace cviai