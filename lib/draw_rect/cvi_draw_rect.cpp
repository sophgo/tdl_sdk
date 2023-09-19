#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "cvi_draw_rect.h"
#include "draw_rect/draw_rect.hpp"

template <typename T>
inline CVI_S32 Lib_DrawRect(const T *meta, VIDEO_FRAME_INFO_S *frame,
                        const bool drawText, cvai_service_brush_t brush) {
  if (meta->size <= 0) return CVIAI_SUCCESS;

  std::vector<cvai_service_brush_t> brushes(meta->size, brush);
  return cviai::service::DrawMeta(meta, frame, drawText, brushes);
}

template <typename T>
inline CVI_S32 Lib_DrawRect(const T *meta, VIDEO_FRAME_INFO_S *frame,
                        const bool drawText, cvai_service_brush_t *brushes) {
  if (meta->size <= 0) return CVIAI_SUCCESS;

  std::vector<cvai_service_brush_t> vec_brushes(brushes, brushes + meta->size);
  return cviai::service::DrawMeta(meta, frame, drawText, vec_brushes);
}

CVI_S32 CVI_AI_FaceDrawRect(const cvai_face_t *meta,
                            VIDEO_FRAME_INFO_S *frame, const bool drawText,
                            cvai_service_brush_t brush) {
  return Lib_DrawRect(meta, frame, drawText, brush);
}

CVI_S32 CVI_AI_FaceDrawRect2(const cvai_face_t *meta,
                            VIDEO_FRAME_INFO_S *frame, const bool drawText,
                            cvai_service_brush_t *brush) {
  return Lib_DrawRect(meta, frame, drawText, brush);
}

CVI_S32 CVI_AI_ObjectDrawRect(const cvai_object_t *meta,
                            VIDEO_FRAME_INFO_S *frame, const bool drawText,
                            cvai_service_brush_t brush) {
  return Lib_DrawRect(meta, frame, drawText, brush);
}

CVI_S32 CVI_AI_ObjectDrawRect2(const cvai_object_t *meta,
                            VIDEO_FRAME_INFO_S *frame, const bool drawText,
                            cvai_service_brush_t *brush) {
  return Lib_DrawRect(meta, frame, drawText, brush);
}
