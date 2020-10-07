#include "es_classification.hpp"
#include "core/cviai_types_mem.h"
#include "opencv2/opencv.hpp"

#define N_FFT 1024
#define WIN_LENGTH 1024
#define HOP_LENGTH 512
#define ESC_OUT_NAME "prob"
namespace cviai {

ESClassification::ESClassification() {
  mp_config = std::make_unique<ModelConfig>();
  hannWindow = cv::Mat_<float>(1, N_FFT, 0.0f);
  float pi = 3.14159265358979323846;
  int insert_cnt = 0;
  if (N_FFT >= WIN_LENGTH) {
    insert_cnt = (N_FFT - WIN_LENGTH) / 2;
  }
  for (int k = 1; k <= WIN_LENGTH; k++) {
    hannWindow(0, k - 1 + insert_cnt) = float(0.5 * (1 - cos(2 * pi * k / (WIN_LENGTH + 1))));
  }
}

ESClassification::~ESClassification() {}

int ESClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  cv::Mat_<float> image(img_height, img_width, 0.0f);
  short *temp_buffer = (short *)stOutFrame->stVFrame.pu8VirAddr[0];
  for (int i = 0; i < img_width; ++i) {
    image.at<float>(0, i) = (float)temp_buffer[i] / 32768.0;  // turn to pcm format
  }
  cv::Mat_<float> mag = STFT(&image);
  mag = cv::abs(mag);
  CVI_TENSOR *input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, mp_input_tensors, m_input_num);
  memcpy((float *)CVI_NN_TensorPtr(input), (float *)mag.data, CVI_NN_TensorSize(input));
  run(stOutFrame);

  CVI_TENSOR *out = CVI_NN_GetTensorByName(ESC_OUT_NAME, mp_output_tensors, m_output_num);
  *index = get_top_k((float *)CVI_NN_TensorPtr(out), CVI_NN_TensorCount(out));
  return CVI_SUCCESS;
}

int ESClassification::get_top_k(float *result, size_t count) {
  int TOP_K = 1;
  float *data = (float *)malloc(count * sizeof(float));
  memcpy(data, result, count * sizeof(float));
  size_t idx = -1;
  for (int k = 0; k < TOP_K; k++) {
    float max = 0;
    for (size_t i = 0; i < count; i++) {
      if (result[i] > max) {
        max = data[i];
        idx = i;
      }
    }
  }
  return idx;
}

cv::Mat_<float> ESClassification::STFT(cv::Mat_<float> *data) {
  int pad_lenght = N_FFT / 2;
  cv::Mat_<float> data_padbuffer;
  cv::copyMakeBorder(*data, data_padbuffer, 0, 0, pad_lenght, pad_lenght, cv::BORDER_REFLECT_101);

  int pad_size = data_padbuffer.rows * data_padbuffer.cols;  // padbuffer.size()
  int number_feature_vectors = (pad_size - N_FFT) / HOP_LENGTH + 1;
  int number_coefficients = N_FFT / 2 + 1;
  cv::Mat_<float> feature_vector(number_feature_vectors, number_coefficients, 0.0f);

  ESCFFT fft;
  fft.init(size_t(N_FFT));
  for (int i = 0; i <= pad_size - N_FFT; i += HOP_LENGTH) {
    cv::Mat_<float> framef = cv::Mat_<float>(1, N_FFT, (float *)(data_padbuffer.data) + i).clone();
    framef = framef.mul(hannWindow);

    cv::Mat_<float> Xrf(1, number_coefficients);
    cv::Mat_<float> Xif(1, number_coefficients);
    fft.fft((float *)(framef.data), (float *)(Xrf.data), (float *)(Xif.data));

    cv::pow(Xrf, 2, Xrf);
    cv::pow(Xif, 2, Xif);
    cv::Mat_<float> cv_feature(1, number_coefficients, &(feature_vector[i / HOP_LENGTH][0]));
    cv::sqrt(Xrf + Xif, cv_feature);
  }
  return feature_vector;
}

}  // namespace cviai
