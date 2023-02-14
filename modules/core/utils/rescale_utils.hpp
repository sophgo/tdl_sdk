#pragma once
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

namespace cviai {

cvai_bbox_t box_rescale_c(const float frame_width, const float frame_height, const float nn_width,
                          const float nn_height, const cvai_bbox_t bbox, float *ratio,
                          float *pad_width, float *pad_height);
cvai_bbox_t box_rescale_rb(const float frame_width, const float frame_height, const float nn_width,
                           const float nn_height, const cvai_bbox_t bbox, float *ratio);

cvai_bbox_t __attribute__((visibility("default")))
box_rescale(const float frame_width, const float frame_height, const float nn_width,
            const float nn_height, const cvai_bbox_t bbox, const meta_rescale_type_e type);

cvai_dms_od_info_t info_rescale_c(const float width, const float height, const float new_width,
                                  const float new_height, const cvai_dms_od_info_t &dms_info);
cvai_face_info_t info_rescale_c(const float width, const float height, const float new_width,
                                const float new_height, const cvai_face_info_t &face_info);
cvai_face_info_t info_rescale_rb(const float width, const float height, const float new_width,
                                 const float new_height, const cvai_face_info_t &face_info);
cvai_face_info_t info_extern_crop_resize_img(const float frame_width, const float frame_height,
                                             const cvai_face_info_t *face_info, int *p_dst_size);
void info_rescale_nocopy_c(const float width, const float height, const float new_width,
                           const float new_height, cvai_face_info_t *face_info);
void info_rescale_nocopy_rb(const float width, const float height, const float new_width,
                            const float new_height, cvai_face_info_t *face_info);
cvai_face_info_t info_rescale_c(const float new_width, const float new_height,
                                const cvai_face_t &face_meta, const int face_idx);
cvai_face_info_t info_rescale_rb(const float new_width, const float new_height,
                                 const cvai_face_t &face_meta, const int face_idx);

}  // namespace cviai
