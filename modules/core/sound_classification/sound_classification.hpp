#include <arm_neon.h>
#include <cvi_comm_vb.h>
#pragma once
#include "ESCFFT.hpp"
#include "core.hpp"
//#include "sound_utils.hpp"

#define FLT_MIN 1.175494351e-38F

namespace cviai {

int borderInterpolate(int p, int len);

void copyMakeBorder(const float *src, float *dst, int srcLen, int top, int left);
// neon structure
struct v_float32x4 {
  typedef float lane_type;
  enum { nlanes = 4 };

  v_float32x4() {}
  explicit v_float32x4(float32x4_t v) : val(v) {}
  v_float32x4(float v0, float v1, float v2, float v3) {
    float v[] = {v0, v1, v2, v3};
    val = vld1q_f32(v);
  }
  float get0() const { return vgetq_lane_f32(val, 0); }
  float32x4_t val;
};

#define HAL_IMPL_NEON_LOADSTORE_OP(_Tpvec, _Tp, suffix)                        \
  inline _Tpvec v_load(const _Tp *ptr) { return _Tpvec(vld1q_##suffix(ptr)); } \
  inline void v_store(_Tp *ptr, const _Tpvec &a) { vst1q_##suffix(ptr, a.val); }

HAL_IMPL_NEON_LOADSTORE_OP(v_float32x4, float, f32)

// neon abs
inline v_float32x4 v_abs(v_float32x4 x) { return v_float32x4(vabsq_f32(x.val)); }

// neon sqrt
inline v_float32x4 v_sqrt(const v_float32x4 &x) {
  float32x4_t x1 = vmaxq_f32(x.val, vdupq_n_f32(FLT_MIN));
  float32x4_t e = vrsqrteq_f32(x1);
  e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, e), e), e);
  e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x1, e), e), e);
  return v_float32x4(vmulq_f32(x.val, e));
}

// Mat structure
class Mat {
 public:
  Mat();
  Mat(int l) {
    this->rows = 1;
    this->cols = l;
    this->len = this->cols;
    this->data = new float[l];
  };

  Mat(int r, int c) {
    this->rows = r;
    this->cols = c;
    this->len = this->cols * this->rows;
    this->data = new float[r * c];
  };

  virtual ~Mat() { delete[] this->data; };

  float *ptr(int r, int c) { return (float *)(this->data + r * this->cols + c); };

  float &at(int index) { return ((float *)(this->data))[index]; };

  float &at(int r, int c) { return ((float *)(this->data + r * this->cols))[c]; };

  void reset() { memset(this->data, 0, sizeof(float) * this->cols * this->rows); };

  void add(Mat *src) {
    for (int i = 0; i < this->len; ++i) {
      this->at(i) += src->at(i);
    }
  };

  void pow() {
    int i = 0;
    for (; i <= this->len - 8; i += 8) {
      float32x4_t v_dst1 = vmulq_f32(vld1q_f32(this->data + i), vld1q_f32(this->data + i));
      float32x4_t v_dst2 = vmulq_f32(vld1q_f32(this->data + i + 4), vld1q_f32(this->data + i + 4));
      vst1q_f32(this->data + i, v_dst1);
      vst1q_f32(this->data + i + 4, v_dst2);
    }
    for (; i < this->len; ++i) {
      this->at(i) = this->at(i) * this->at(i);
    }
  };

  void multipy(Mat *src) {
    int i = 0;
    for (; i <= this->len - 8; i += 8) {
      float32x4_t v_dst1 = vmulq_f32(vld1q_f32(this->data + i), vld1q_f32(src->data + i));
      float32x4_t v_dst2 = vmulq_f32(vld1q_f32(this->data + i + 4), vld1q_f32(src->data + i + 4));
      vst1q_f32(this->data + i, v_dst1);
      vst1q_f32(this->data + i + 4, v_dst2);
    }
    for (; i < this->len; ++i) {
      this->at(i) = this->at(i) * src->at(i);
    }
  };

  void sqrt() {
    const int VECSZ = v_float32x4::nlanes;
    int step = VECSZ * 2;
    int i;
    for (i = 0; i <= this->len - step; i += step) {
      v_float32x4 t0 = v_load(this->data + i), t1 = v_load(this->data + i + VECSZ);
      t0 = v_abs(v_sqrt(t0));
      t1 = v_abs(v_sqrt(t1));
      v_store(this->data + i, t0);
      v_store(this->data + i + VECSZ, t1);
    }
    for (; i < len; ++i) {
      this->at(i) = fabs(std::sqrt(this->at(i)));
    }
  };

  int rows;
  int cols;
  int len;
  float *data;
};

class SoundClassification final : public Core {
 public:
  SoundClassification();
  virtual ~SoundClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index);
  void prepareInputTensor(std::vector<Mat *> &input_mat);

 private:
  float *STFT(float *data, int channel, int img_height, int img_width);
  void STFT(Mat *data, int channel, Mat *feature_vector);
  int get_top_k(float *result, size_t count);
  int hop_length[1] = {256};
  int win_length[1] = {1024};
  int feat_width = 513;
  int feat_height = 188;
  int Channel = 1;
  int pad_length;
  int pad_size;
  Mat *framef;
  Mat *Xrf;
  Mat *Xif;
  Mat *data_padbuffer;
  std::vector<Mat *> feature_vectors;
  std::vector<Mat *> hannWindows;
  ESCFFT fft;
};
}  // namespace cviai
