#pragma once
#include "core/core/cvai_core_types.h"
#include "face_utils.hpp"

namespace cviai {

uint32_t get_image_size(cvai_image_t *dst);

int crop_image(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_bbox_t *bbox);

int crop_image_face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst, cvai_face_info_t *face_info,
                    bool align);

}  // namespace cviai