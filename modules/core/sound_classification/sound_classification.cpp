#include "sound_classification.hpp"
#include "core/core/cvai_errno.h"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "cviai_trace.hpp"

#define N_FFT 1024
#define ESC_OUT_NAME "prob_dequant"

namespace cviai {

SoundClassification::SoundClassification() : Core(CVI_MEM_SYSTEM) {
  int insert_cnt = 0;
  float pi = 3.14159265358979323846;
  // Calculate 3 different stft hannwindow
  for (int i = 0; i < Channel; ++i) {
    hannWindow[i] = cv::Mat(1, N_FFT, CV_32F, 0.0f);
    if (N_FFT >= win_length[i]) {
      insert_cnt = (N_FFT - win_length[i]) / 2;
    }
    for (int k = 1; k <= win_length[i]; ++k) {
      hannWindow[i].at<float>(0, k - 1 + insert_cnt) =
          static_cast<float>(0.5 * (1 - cos(2 * pi * k / (win_length[i] + 1))));
    }
  }
  pad_length = N_FFT / 2;
  number_coefficients = N_FFT / 2 + 1;
  // init fft
  fft.init(size_t(N_FFT));
}

SoundClassification::~SoundClassification() {}

int SoundClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  cv::Mat image(img_height, img_width, CV_32F, 0.0f);

  // save audio to image array
  short *temp_buffer = reinterpret_cast<short *>(stOutFrame->stVFrame.pu8VirAddr[0]);
  for (int i = 0; i < img_width; ++i) {
    image.at<float>(0, i) = static_cast<float>(temp_buffer[i] / 32768.0);  // turn to pcm format
  }

  // 1 channel input with different stft
  std::vector<cv::Mat> input;
  for (int i = 0; i < Channel; ++i) {
    cv::Mat mag;
    mag = STFT(&image, i);
    mag = cv::abs(mag);
    cv::resize(mag, mag, cv::Size(feat_width, feat_height), 0, 0, cv::INTER_LINEAR);
    input.push_back(mag);
  }

  prepareInputTensor(input);

  std::vector<VIDEO_FRAME_INFO_S *> frames = {stOutFrame};
  int ret = run(frames);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }

  const TensorInfo &info = getOutputTensorInfo(ESC_OUT_NAME);

  // get top k
  *index = get_top_k(info.get<float>(), info.tensor_elem);
  return CVIAI_SUCCESS;
}

int SoundClassification::get_top_k(float *result, size_t count) {
  int TOP_K = 1;
  float *data = reinterpret_cast<float *>(malloc(count * sizeof(float)));
  memcpy(data, result, count * sizeof(float));
  int idx = -1;
  float pct = 0.0;
  for (int k = 0; k < TOP_K; k++) {
    float max = 0;
    for (size_t i = 0; i < count; i++) {
      if (result[i] > max) {
        max = data[i];
        idx = i;
      }
    }
    pct = max;
  }
  if (pct < 0.6) return count;  // Office
  return idx;
}

cv::Mat SoundClassification::STFT(cv::Mat *data, int channel) {
  cv::Mat data_padbuffer;
  cv::copyMakeBorder(*data, data_padbuffer, 0, 0, pad_length, pad_length, cv::BORDER_REFLECT_101);
  int pad_size = data_padbuffer.rows * data_padbuffer.cols;  // padbuffer.size()
  int number_feature_vectors = (pad_size - N_FFT) / hop_length[channel] + 1;
  cv::Mat feature_vector(number_feature_vectors, number_coefficients, CV_32F, 0.0f);
  for (int i = 0; i <= pad_size - N_FFT; i += hop_length[channel]) {
    cv::Mat framef(1, N_FFT, CV_32F, 0.0f);
    memcpy(framef.data, data_padbuffer.ptr<float>(0, i), sizeof(float) * N_FFT);
    framef = framef.mul(hannWindow[channel]);

    cv::Mat Xrf(1, number_coefficients, CV_32F, 0.0f);
    cv::Mat Xif(1, number_coefficients, CV_32F, 0.0f);
    fft.fft(reinterpret_cast<float *>(framef.data), reinterpret_cast<float *>(Xrf.data),
            reinterpret_cast<float *>(Xif.data));

    cv::pow(Xrf, 2, Xrf);
    cv::pow(Xif, 2, Xif);

    cv::Mat cv_feature(1, number_coefficients, CV_32F, 0.0f);
    cv_feature.data =
        reinterpret_cast<unsigned char *>(feature_vector.ptr<float>(i / hop_length[channel], 0));

    cv::sqrt(Xrf + Xif, cv_feature);
  }
  return feature_vector;
}

void SoundClassification::prepareInputTensor(std::vector<cv::Mat> &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  float *input_ptr = tinfo.get<float>();

  for (int c = 0; c < Channel; ++c) {
    int size = input_mat[c].rows * input_mat[c].cols;
    for (int r = 0; r < input_mat[c].rows; ++r) {
      memcpy(input_ptr + size * c + input_mat[c].cols * r, input_mat[c].ptr(r, 0),
             input_mat[c].cols * sizeof(float));
    }
  }
}
}  // namespace cviai
