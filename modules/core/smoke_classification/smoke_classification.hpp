#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"
#include "opencv2/core.hpp"

namespace cviai {

class SmokeClassification final : public Core {
 public:
  SmokeClassification();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);
};
}  // namespace cviai
