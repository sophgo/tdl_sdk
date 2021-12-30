#include "sound_classification.hpp"
#include <cstring>
#define N_FFT 1024
#define ESC_OUT_NAME "prob_dequant"

namespace cviai {

int borderInterpolate(int p, int len) {
  int delta = 1;
  if (len == 1) return 0;
  do {
    if (p < 0)
      p = -p - 1 + delta;
    else
      p = len - 1 - (p - len) - delta;
  } while ((unsigned)p >= (unsigned)len);
  return p;
}

void copyMakeBorder(const float *src, float *dst, int srcLen, int top, int left) {
  int i, j;
  int *tab = new int[left * 2];
  int right = left;
  for (i = 0; i < left; i++) {
    j = borderInterpolate(i - left, srcLen);
    tab[i] = j;
  }
  for (i = 0; i < right; i++) {
    j = borderInterpolate(srcLen + i, srcLen);
    tab[(i + left)] = j;
  }
  float *dstInner = (dst + left);
  memcpy(dstInner, src, srcLen * sizeof(float));
  for (i = 0; i < left; ++i) {
    dstInner[i - left] = src[tab[i]];
  }
  for (i = 0; i < right; ++i) {
    dstInner[i + srcLen] = src[tab[i + left]];
  }
}

SoundClassification::SoundClassification() : Core(CVI_MEM_SYSTEM) {
  int insert_cnt = 0;
  float pi = 3.14159265358979323846;

  // Calculate 3 different stft hannwindow
  for (int i = 0; i < Channel; ++i) {
    Mat *hannWindow = new Mat(N_FFT);
    hannWindow->reset();
    if (N_FFT >= win_length[i]) {
      insert_cnt = (N_FFT - win_length[i]) / 2;
    }
    for (int k = 1; k <= win_length[i]; ++k) {
      hannWindow->at(k - 1 + insert_cnt) =
          static_cast<float>(0.5 * (1 - cos(2 * pi * k / (win_length[i] + 1))));
    }
    hannWindows.push_back(hannWindow);
  }
  pad_length = N_FFT / 2;
  // init fft
  fft.init(size_t(N_FFT));
  framef = new Mat(N_FFT);
  Xrf = new Mat(feat_width);
  Xif = new Mat(feat_width);
  data_padbuffer = nullptr;
  for (int i = 0; i < Channel; ++i) {
    Mat *feature_vector = new Mat(feat_height, feat_width);
    feature_vectors.push_back(feature_vector);
  }
}

SoundClassification::~SoundClassification() {
  delete framef;
  delete Xrf;
  delete Xif;
  delete data_padbuffer;
  for (int i = 0; i < Channel; ++i) delete feature_vectors[i];
}

int SoundClassification::inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index) {
  int img_width = stOutFrame->stVFrame.u32Width / 2;  // unit: 16 bits
  int img_height = stOutFrame->stVFrame.u32Height;
  Mat *image = new Mat(img_height, img_width);

  // save audio to image array
  short *temp_buffer = reinterpret_cast<short *>(stOutFrame->stVFrame.pu8VirAddr[0]);

  for (int i = 0; i < img_height; ++i) {
    for (int j = 0; j < img_width; ++j) {
      image->at(i, j) = static_cast<float>(temp_buffer[i * img_width + j] / 32768.0);
    }
  }

  // 1 channel input with different stft
  pad_size = img_height * (img_width + pad_length + pad_length);
  if (data_padbuffer == nullptr) data_padbuffer = new Mat(pad_size);

  for (int i = 0; i < Channel; ++i) {
    feature_vectors[i]->reset();
    STFT(image, i, feature_vectors[i]);
  }

  prepareInputTensor(feature_vectors);
  std::vector<VIDEO_FRAME_INFO_S *> frames = {stOutFrame};
  run(frames);

  const TensorInfo &info = getOutputTensorInfo(ESC_OUT_NAME);

  // get top k
  *index = get_top_k(info.get<float>(), info.tensor_elem);
  delete image;
  return CVI_SUCCESS;
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

void SoundClassification::STFT(Mat *data, int channel, Mat *feature_vector) {
  int len = data->cols * data->rows;
  data_padbuffer->reset();

  copyMakeBorder(data->data, data_padbuffer->data, len, 0, pad_length);
  int count = 0;
  for (int i = 0; i <= pad_size - N_FFT; i += hop_length[channel]) {
    framef->reset();

    memcpy(framef->data, data_padbuffer->ptr(0, i), sizeof(float) * N_FFT);
    framef->multipy(hannWindows[channel]);

    Xrf->reset();
    Xif->reset();

    fft.fft(reinterpret_cast<float *>(framef->data), reinterpret_cast<float *>(Xrf->data),
            reinterpret_cast<float *>(Xif->data));

    Xrf->pow();
    Xif->pow();
    Xrf->add(Xif);
    Xrf->sqrt();
    memcpy(feature_vector->ptr(count++, 0), Xrf->data, sizeof(float) * feat_width);
  }
}

void SoundClassification::prepareInputTensor(std::vector<Mat *> &input_mat) {
  const TensorInfo &tinfo = getInputTensorInfo(0);
  float *input_ptr = tinfo.get<float>();

  for (int c = 0; c < Channel; ++c) {
    int size = input_mat[c]->rows * input_mat[c]->cols;
    memcpy(input_ptr + c * size, input_mat[c]->data, size * sizeof(float));
  }
}
}  // namespace cviai
