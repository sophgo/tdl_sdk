#pragma once
#include "ESCFFT.hpp"
#include "core.hpp"
#include "core/face/cvai_face_types.h"
#include "opencv2/opencv.hpp"

namespace cviai {

class ESClassification final : public Core {
 public:
  ESClassification();
  virtual ~ESClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index);

 private:
  cv::Mat_<float> hannWindow;
  cv::Mat_<float> STFT(cv::Mat_<float> *data);
  int get_top_k(float *result, size_t count);
};
}  // namespace cviai
