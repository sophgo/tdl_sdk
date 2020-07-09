#ifndef _DRAW_UTILS_H_
#define _DRAW_UTILS_H_

#include "cv183x_facelib_v0.0.1.h"

void DrawFaceMeta(VIDEO_FRAME_INFO_S *draw_frame, cvi_face_t *face_meta);
void DrawObjMeta(VIDEO_FRAME_INFO_S *draw_frame, cvi_object_meta_t *meta);

#endif
