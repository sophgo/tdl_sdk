#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class FaceLandmarker final : public Core {
 public:
  FaceLandmarker();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);
  void Preprocessing(cvai_face_info_t *face_info, int *max_side, int img_width, int img_height);
};
}  // namespace cviai
