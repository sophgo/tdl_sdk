#pragma once
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#ifdef MARS
#include <linux/cvi_comm_video.h>
#else
#include <cvi_comm_video.h>
#endif
#ifdef USE_IVE
#include "ive/cvi_draw_ive.h"
#endif
#include "service/cviai_service_types.h"

#define DEFAULT_RECT_COLOR_R (53. / 255.)
#define DEFAULT_RECT_COLOR_G (208. / 255.)
#define DEFAULT_RECT_COLOR_B (217. / 255.)
#define DEFAULT_RECT_THICKNESS 4
#define DEFAULT_TEXT_THICKNESS 1
#define DEFAULT_RADIUS 1

namespace cviai {
namespace service {

typedef struct {
  float r;
  float g;
  float b;
} color_rgb;

void DrawRect(VIDEO_FRAME_INFO_S *frame, float x1, float x2, float y1, float y2, const char *name,
              color_rgb color, int rect_thinkness, const bool draw_text);

int _WriteText(VIDEO_FRAME_INFO_S *frame, int x, int y, const char *name, color_rgb color,
               int thinkness);

int WriteText(char *name, int x, int y, VIDEO_FRAME_INFO_S *drawFrame, float r, float g, float b);

template <typename T>
int DrawMeta(const T *meta, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText,
             cvai_service_brush_t brush);

#ifdef USE_IVE
template <typename T>
int DrawMetaIVE(const T *meta, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText,
                IVE_DRAW_RECT_CTRL *pstDrawRectCtrl);

template <typename T>
void getDrawRectCTRL(const T *meta, VIDEO_FRAME_INFO_S *drawFrame,
                     IVE_DRAW_RECT_CTRL *pstDrawRectCtrl, IVE_COLOR_S color);
#endif

int DrawPose17(const cvai_object_t *obj, VIDEO_FRAME_INFO_S *frame);

int DrawPts(cvai_pts_t *pts, VIDEO_FRAME_INFO_S *drawFrame);

void _DrawPts(VIDEO_FRAME_INFO_S *frame, cvai_pts_t *pts, color_rgb color, int raduis);

int Draw5Landmark(const cvai_face_t *meta, VIDEO_FRAME_INFO_S *frame);

int DrawPolygon(VIDEO_FRAME_INFO_S *frame, const cvai_pts_t *pts, cvai_service_brush_t brush);

}  // namespace service
}  // namespace cviai
