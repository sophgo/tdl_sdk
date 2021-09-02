#include "core/cviai_rescale_bbox.h"
#include "core/core/cvai_errno.h"
#include "utils/core_utils.hpp"
#include "utils/face_utils.hpp"

CVI_S32 CVI_AI_RescaleMetaCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  if (frame->stVFrame.u32Width == face->width && frame->stVFrame.u32Height == face->height) {
    return CVIAI_SUCCESS;
  }
  for (uint32_t i = 0; i < face->size; i++) {
    cviai::info_rescale_nocopy_c(face->width, face->height, frame->stVFrame.u32Width,
                                 frame->stVFrame.u32Height, &face->info[i]);
  }
  face->width = frame->stVFrame.u32Width;
  face->height = frame->stVFrame.u32Height;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_RescaleMetaCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  if (frame->stVFrame.u32Width == obj->width && frame->stVFrame.u32Height == obj->height) {
    return CVIAI_SUCCESS;
  }
  float ratio, pad_width, pad_height;
  for (uint32_t i = 0; i < obj->size; i++) {
    obj->info[i].bbox =
        cviai::box_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height, obj->width,
                             obj->height, obj->info[i].bbox, &ratio, &pad_width, &pad_height);
  }
  obj->width = frame->stVFrame.u32Width;
  obj->height = frame->stVFrame.u32Height;
  return CVIAI_SUCCESS;
}
CVI_S32 CVI_AI_RescaleMetaRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  if (frame->stVFrame.u32Width == face->width && frame->stVFrame.u32Height == face->height) {
    return CVIAI_SUCCESS;
  }
  for (uint32_t i = 0; i < face->size; i++) {
    cviai::info_rescale_nocopy_rb(face->width, face->height, frame->stVFrame.u32Width,
                                  frame->stVFrame.u32Height, &face->info[i]);
  }
  face->width = frame->stVFrame.u32Width;
  face->height = frame->stVFrame.u32Height;
  return CVIAI_SUCCESS;
}
CVI_S32 CVI_AI_RescaleMetaRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  if (frame->stVFrame.u32Width == obj->width && frame->stVFrame.u32Height == obj->height) {
    return CVIAI_SUCCESS;
  }
  float ratio;
  for (uint32_t i = 0; i < obj->size; i++) {
    obj->info[i].bbox = cviai::box_rescale_rb(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                              obj->width, obj->height, obj->info[i].bbox, &ratio);
  }
  obj->width = frame->stVFrame.u32Width;
  obj->height = frame->stVFrame.u32Height;
  return CVIAI_SUCCESS;
}

CVI_S32 CVI_AI_RescaleMetaCenterFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  return CVI_AI_RescaleMetaCenterCpp(frame, face);
}
CVI_S32 CVI_AI_RescaleMetaCenterObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  return CVI_AI_RescaleMetaCenterCpp(frame, obj);
}
CVI_S32 CVI_AI_RescaleMetaRBFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  return CVI_AI_RescaleMetaRBCpp(frame, face);
}
CVI_S32 CVI_AI_RescaleMetaRBObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  return CVI_AI_RescaleMetaRBCpp(frame, obj);
}
