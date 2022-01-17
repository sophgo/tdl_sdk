#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"
#include "opencv2/core.hpp"

namespace cviai {

class FaceLandmarker final : public Core {
 public:
  FaceLandmarker();
  virtual ~FaceLandmarker();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void prepareInputTensor(cv::Mat &input_mat);
  void Preprocessing(cvai_face_info_t *face_info, int *max_side, int img_width, int img_height);

  int landmark_num = 106;
};
}  // namespace cviai
