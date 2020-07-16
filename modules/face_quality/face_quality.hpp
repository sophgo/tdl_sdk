#pragma once
#include "core.hpp"
#include "face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class FaceQuality final : public Core {
 public:
  FaceQuality();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta);
};
}  // namespace cviai
