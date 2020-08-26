#include "core/cviai_rescale_bbox.h"
#include "utils/core_utils.hpp"

CVI_S32 CVI_AI_RescaleBBoxCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  if (frame->stVFrame.u32Width == face->width && frame->stVFrame.u32Height == face->height) {
    return CVI_SUCCESS;
  }
  for (uint32_t i = 0; i < face->size; i++) {
    face->info[i].bbox = cviai::box_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                              face->width, face->height, face->info[i].bbox);
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_RescaleBBoxCenterCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  if (frame->stVFrame.u32Width == obj->width && frame->stVFrame.u32Height == obj->height) {
    return CVI_SUCCESS;
  }
  for (uint32_t i = 0; i < obj->size; i++) {
    obj->info[i].bbox = cviai::box_rescale_c(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                             obj->width, obj->height, obj->info[i].bbox);
  }
  return CVI_SUCCESS;
}
CVI_S32 CVI_AI_RescaleBBoxRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  if (frame->stVFrame.u32Width == face->width && frame->stVFrame.u32Height == face->height) {
    return CVI_SUCCESS;
  }
  for (uint32_t i = 0; i < face->size; i++) {
    face->info[i].bbox = cviai::box_rescale_rb(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                               face->width, face->height, face->info[i].bbox);
  }
  return CVI_SUCCESS;
}
CVI_S32 CVI_AI_RescaleBBoxRBCpp(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  if (frame->stVFrame.u32Width == obj->width && frame->stVFrame.u32Height == obj->height) {
    return CVI_SUCCESS;
  }
  for (uint32_t i = 0; i < obj->size; i++) {
    obj->info[i].bbox = cviai::box_rescale_rb(frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                                              obj->width, obj->height, obj->info[i].bbox);
  }
  return CVI_SUCCESS;
}

CVI_S32 CVI_AI_RescaleBBoxCenterFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  return CVI_AI_RescaleBBoxCenterCpp(frame, face);
}
CVI_S32 CVI_AI_RescaleBBoxCenterObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  return CVI_AI_RescaleBBoxCenterCpp(frame, obj);
}
CVI_S32 CVI_AI_RescaleBBoxRBFace(const VIDEO_FRAME_INFO_S *frame, cvai_face_t *face) {
  return CVI_AI_RescaleBBoxRBCpp(frame, face);
}
CVI_S32 CVI_AI_RescaleBBoxRBObj(const VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj) {
  return CVI_AI_RescaleBBoxRBCpp(frame, obj);
}
