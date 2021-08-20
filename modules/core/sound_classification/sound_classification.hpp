#pragma once
#include "ESCFFT.hpp"
#include "core.hpp"
#include "opencv2/opencv.hpp"

namespace cviai {

class SoundClassification final : public Core {
 public:
  SoundClassification();
  virtual ~SoundClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index);
  void prepareInputTensor(std::vector<cv::Mat> &input_mat);

 private:
  cv::Mat STFT(cv::Mat *data, int channel);
  int get_top_k(float *result, size_t count);
  int pad_length;
  int number_coefficients;
  int hop_length[1] = {256};
  int win_length[1] = {1024};
  int feat_width = 513;
  int feat_height = 188;
  int Channel = 1;
  cv::Mat hannWindow[1];
  ESCFFT fft;
};
}  // namespace cviai
