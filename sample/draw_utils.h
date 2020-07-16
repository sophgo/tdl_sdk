#ifndef _DRAW_UTILS_H_
#define _DRAW_UTILS_H_

#include "cviai.h"

void DrawFaceMeta(VIDEO_FRAME_INFO_S *draw_frame, cvai_face_t *face_meta);
void DrawObjMeta(VIDEO_FRAME_INFO_S *draw_frame, cvai_object_t *meta);

#endif
