#include "es_classification.hpp"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"
#include "opencv2/opencv.hpp"

#define N_FFT 1024
#define ESC_OUT_NAME "prob"
#define PI 3.14159265358979323846
namespace cviai {

ESClassification::ESClassification() : Core() {
  int insert_cnt = 0;
  // Calculate 3 different stft hannwindow
  for (int i = 0; i < 3; ++i) {
    hannWindow[i] = cv::Mat_<float>(1, N_FFT, 0.0f);
    if (N_FFT >= win_length[i]) {
      insert_cnt = (N_FFT - win_length[i]) / 2;
    }
    for (int k = 1; k <= win_length[i]; ++k) {
      hannWindow[i](0, k - 1 + insert_cnt) =
          float(0.5 * (1 - cos(2 * PI * k / (win_length[i] + 1))));
    }
  }
  pad_length = N_FFT / 2;
  number_coefficients = N_FFT / 2 + 1;
  // init fft
  fft.init(size_t(N_FFT));
}

ESClassification::~ESClassification() {}

int ESClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  cv::Mat_<float> image(img_height, img_width, 0.0f);
  // save audio to image array
  short *temp_buffer = (short *)stOutFrame->stVFrame.pu8VirAddr[0];
  for (int i = 0; i < img_width; ++i) {
    image.at<float>(0, i) = (float)temp_buffer[i] / 32768.0;  // turn to pcm format
  }
  // 3 channel input with different stft
  cv::Mat_<float> mag[3];
  for (int i = 0; i < 3; ++i) {
    mag[i] = STFT(&image, i);
    mag[i] = cv::abs(mag[i]);
    cv::resize(mag[i], mag[i], cv::Size(feat_width, feat_height), 0, 0, cv::INTER_LINEAR);
  }

  uint16_t *input_ptr = getInputRawPtr<uint16_t>(0);
  int size = feat_height * feat_width;
  for (int i = 0; i < 3; ++i) {
    for (int r = 0; r < feat_height; ++r) {
      for (int c = 0; c < feat_width; ++c) {
        uint16_t bf16_input = 0;
        floatToBF16((float *)mag[i].ptr(r, c), &bf16_input);
        memcpy(input_ptr + i * size + mag[i].cols * r + c, &bf16_input, sizeof(uint16_t));
      }
    }
  }

  std::vector<VIDEO_FRAME_INFO_S *> frames = {stOutFrame};
  run(frames);

  const TensorInfo &info = getOutputTensorInfo(ESC_OUT_NAME);
  // get top k
  *index = get_top_k(info.get<float>(), info.tensor_elem);
  return CVI_SUCCESS;
}

int ESClassification::get_top_k(float *result, size_t count) {
  int TOP_K = 1;
  float *data = (float *)malloc(count * sizeof(float));
  memcpy(data, result, count * sizeof(float));
  size_t idx = -1;
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
  if (pct < 0.6) return 7;  // Office
  return idx;
}

cv::Mat_<float> ESClassification::STFT(cv::Mat_<float> *data, int channel) {
  cv::Mat_<float> data_padbuffer;
  cv::copyMakeBorder(*data, data_padbuffer, 0, 0, pad_length, pad_length, cv::BORDER_REFLECT_101);
  int pad_size = data_padbuffer.rows * data_padbuffer.cols;  // padbuffer.size()
  int number_feature_vectors = (pad_size - N_FFT) / hop_length[channel] + 1;
  cv::Mat_<float> feature_vector(number_feature_vectors, number_coefficients, 0.0f);
  for (int i = 0; i <= pad_size - N_FFT; i += hop_length[channel]) {
    cv::Mat_<float> framef = cv::Mat_<float>(1, N_FFT, (float *)(data_padbuffer.data) + i).clone();
    framef = framef.mul(hannWindow[channel]);

    cv::Mat_<float> Xrf(1, number_coefficients);
    cv::Mat_<float> Xif(1, number_coefficients);
    fft.fft((float *)(framef.data), (float *)(Xrf.data), (float *)(Xif.data));

    cv::pow(Xrf, 2, Xrf);
    cv::pow(Xif, 2, Xif);
    cv::Mat_<float> cv_feature(1, number_coefficients,
                               &(feature_vector[i / hop_length[channel]][0]));
    cv::sqrt(Xrf + Xif, cv_feature);
  }
  return feature_vector;
}

}  // namespace cviai
