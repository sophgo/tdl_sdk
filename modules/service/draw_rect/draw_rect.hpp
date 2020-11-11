#pragma once
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_video.h>

#define DEFAULT_RECT_COLOR_R (53. / 255.)
#define DEFAULT_RECT_COLOR_G (208. / 255.)
#define DEFAULT_RECT_COLOR_B (217. / 255.)
#define DEFAULT_RECT_THINKNESS 4

namespace cviai {
namespace service {

typedef struct {
  float r;
  float g;
  float b;
} color_rgb;

void DrawRect(VIDEO_FRAME_INFO_S *frame, float x1, float x2, float y1, float y2, const char *name,
              color_rgb color, int rect_thinkness, const bool draw_text);

template <typename T>
int DrawMeta(const T *meta, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText);
}  // namespace service
}  // namespace cviai