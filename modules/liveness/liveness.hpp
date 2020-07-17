#pragma once
#include "core.hpp"
#include "face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class Liveness final : public Core {
 public:
  Liveness(cvai_liveness_ir_position_e ir_pos);
  int inference(VIDEO_FRAME_INFO_S *rgbFrame, VIDEO_FRAME_INFO_S *irFrame, cvai_face_t *meta);

 private:
  void prepareInputTensor(std::vector<cv::Mat> &input_mat);

  cvai_liveness_ir_position_e m_ir_pos;
};
}  // namespace cviai