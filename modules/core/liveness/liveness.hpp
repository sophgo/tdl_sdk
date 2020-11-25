#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class Liveness final : public Core {
 public:
  Liveness();
  int inference(VIDEO_FRAME_INFO_S *rgbFrame, VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *rgb_meta,
                cvai_face_t *ir_meta);

 private:
  void prepareInputTensor(std::vector<cv::Mat> &input_mat);
};
}  // namespace cviai