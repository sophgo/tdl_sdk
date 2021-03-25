#pragma once
#include "ESCFFT.hpp"
#include "core.hpp"
#include "opencv2/opencv.hpp"

namespace cviai {

class ESClassification final : public Core {
 public:
  ESClassification();
  virtual ~ESClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index);

 private:
  ESCFFT fft;
  int pad_length;
  int number_coefficients;
  int hop_length[3] = {144, 224, 256};
  int win_length[3] = {192, 240, 256};
  cv::Mat_<float> hannWindow[3];
  cv::Mat_<float> STFT(cv::Mat_<float> *data, int channel);
  int get_top_k(float *result, size_t count);
  int feat_width = 129;
  int feat_height = 201;
};
}  // namespace cviai
