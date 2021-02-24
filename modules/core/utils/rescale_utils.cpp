#include "rescale_utils.hpp"

#include "core/cviai_types_mem_internal.h"

#include <algorithm>

namespace cviai {
cvai_bbox_t box_rescale_c(const float frame_width, const float frame_height, const float nn_width,
                          const float nn_height, const cvai_bbox_t bbox, float *ratio,
                          float *pad_width, float *pad_height) {
  float ratio_height = (nn_height / frame_height);
  float ratio_width = (nn_width / frame_width);
  if (ratio_height > ratio_width) {
    *ratio = 1.0 / ratio_width;
    *pad_width = 0;
    *pad_height = (nn_height - frame_height * ratio_width) / 2;
  } else {
    *ratio = 1.0 / ratio_height;
    *pad_width = (nn_width - frame_width * ratio_height) / 2;
    *pad_height = 0;
  }

  float x1 = (bbox.x1 - (*pad_width)) * (*ratio);
  float x2 = (bbox.x2 - (*pad_width)) * (*ratio);
  float y1 = (bbox.y1 - (*pad_height)) * (*ratio);
  float y2 = (bbox.y2 - (*pad_height)) * (*ratio);
  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale_rb(const float frame_width, const float frame_height, const float nn_width,
                           const float nn_height, const cvai_bbox_t bbox, float *ratio) {
  float ratio_height = (nn_height / frame_height);
  float ratio_width = (nn_width / frame_width);
  *ratio = 1.0 / std::min(ratio_height, ratio_width);

  float x1 = bbox.x1 * (*ratio);
  float x2 = bbox.x2 * (*ratio);
  float y1 = bbox.y1 * (*ratio);
  float y2 = bbox.y2 * (*ratio);

  cvai_bbox_t new_bbox;
  new_bbox.score = bbox.score;
  new_bbox.x1 = std::max(std::min(x1, (float)(frame_width - 1)), (float)0);
  new_bbox.x2 = std::max(std::min(x2, (float)(frame_width - 1)), (float)0);
  new_bbox.y1 = std::max(std::min(y1, (float)(frame_height - 1)), (float)0);
  new_bbox.y2 = std::max(std::min(y2, (float)(frame_height - 1)), (float)0);
  return new_bbox;
}

cvai_bbox_t box_rescale(const float frame_width, const float frame_height, const float nn_width,
                        const float nn_height, const cvai_bbox_t bbox,
                        const meta_rescale_type_e type) {
  if (frame_width == nn_width && frame_height == nn_height) {
    return bbox;
  }
  float ratio;
  switch (type) {
    case meta_rescale_type_e::RESCALE_CENTER: {
      float pad_width, pad_height;
      return box_rescale_c(frame_width, frame_height, nn_width, nn_height, bbox, &ratio, &pad_width,
                           &pad_height);
    } break;
    case meta_rescale_type_e::RESCALE_RB: {
      return box_rescale_rb(frame_width, frame_height, nn_width, nn_height, bbox, &ratio);
    } break;
    default: {
      // TODO: Add no aspect ratio.
      // For others we'll return the original box.
      LOGW("Unsupported rescaling method %u. Exporting original box.\n", type);
    } break;
  }
  return bbox;
}

cvai_face_info_t info_rescale_c(const float width, const float height, const float new_width,
                                const float new_height, const cvai_face_info_t &face_info) {
  cvai_face_info_t face_info_new;
  CVI_AI_CopyInfoCpp(&face_info, &face_info_new);

  float ratio, pad_width, pad_height;
  face_info_new.bbox = box_rescale_c(new_width, new_height, width, height, face_info.bbox, &ratio,
                                     &pad_width, &pad_height);
  CVI_AI_MemAlloc(face_info.pts.size, &face_info_new.pts);
  for (uint32_t j = 0; j < face_info_new.pts.size; ++j) {
    face_info_new.pts.x[j] = (face_info.pts.x[j] - pad_width) * ratio;
    face_info_new.pts.y[j] = (face_info.pts.y[j] - pad_height) * ratio;
  }
  return face_info_new;
}

cvai_dms_od_info_t info_rescale_c(const float width, const float height, const float new_width,
                                  const float new_height, const cvai_dms_od_info_t &dms_info) {
  cvai_dms_od_info_t dms_info_new;
  CVI_AI_CopyInfoCpp(&dms_info, &dms_info_new);

  float ratio, pad_width, pad_height;
  dms_info_new.bbox = box_rescale_c(new_width, new_height, width, height, dms_info.bbox, &ratio,
                                    &pad_width, &pad_height);
  return dms_info_new;
}
cvai_face_info_t info_rescale_rb(const float width, const float height, const float new_width,
                                 const float new_height, const cvai_face_info_t &face_info) {
  cvai_face_info_t face_info_new;
  CVI_AI_CopyInfoCpp(&face_info, &face_info_new);

  float ratio;
  face_info_new.bbox = box_rescale_rb(new_width, new_height, width, height, face_info.bbox, &ratio);
  for (uint32_t j = 0; j < face_info_new.pts.size; ++j) {
    face_info_new.pts.x[j] = face_info.pts.x[j] * ratio;
    face_info_new.pts.y[j] = face_info.pts.y[j] * ratio;
  }
  return face_info_new;
}

void info_rescale_nocopy_c(const float width, const float height, const float new_width,
                           const float new_height, cvai_face_info_t *face_info) {
  float ratio, pad_width, pad_height;
  face_info->bbox = box_rescale_c(new_width, new_height, width, height, face_info->bbox, &ratio,
                                  &pad_width, &pad_height);
  for (uint32_t j = 0; j < face_info->pts.size; ++j) {
    face_info->pts.x[j] = (face_info->pts.x[j] - pad_width) * ratio;
    face_info->pts.y[j] = (face_info->pts.y[j] - pad_height) * ratio;
  }
}

void info_rescale_nocopy_rb(const float width, const float height, const float new_width,
                            const float new_height, cvai_face_info_t *face_info) {
  float ratio;
  face_info->bbox = box_rescale_rb(new_width, new_height, width, height, face_info->bbox, &ratio);
  for (uint32_t j = 0; j < face_info->pts.size; ++j) {
    face_info->pts.x[j] = face_info->pts.x[j] * ratio;
    face_info->pts.y[j] = face_info->pts.y[j] * ratio;
  }
}

cvai_face_info_t info_rescale_c(const float new_width, const float new_height,
                                const cvai_face_t &face_meta, const int face_idx) {
  return info_rescale_c(face_meta.width, face_meta.height, new_width, new_height,
                        face_meta.info[face_idx]);
}

cvai_face_info_t info_rescale_rb(const float new_width, const float new_height,
                                 const cvai_face_t &face_meta, const int face_idx) {
  return info_rescale_rb(face_meta.width, face_meta.height, new_width, new_height,
                         face_meta.info[face_idx]);
}

}  // namespace cviai
