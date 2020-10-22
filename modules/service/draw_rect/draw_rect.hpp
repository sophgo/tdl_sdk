#pragma once
#include "core/face/cvai_face_types.h"
#include "core/object/cvai_object_types.h"

#include <cvi_comm_video.h>

namespace cviai {
namespace service {
template <typename T>
int DrawMeta(const T *meta, VIDEO_FRAME_INFO_S *drawFrame, const bool drawText);
}
}  // namespace cviai