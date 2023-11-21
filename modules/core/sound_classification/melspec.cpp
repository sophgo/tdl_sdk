#include "melspec.hpp"
#include <iostream>
#include "cvi_tdl_log.hpp"
#include "unsupported/Eigen/FFT"
using namespace melspec;

static Matrixf melfilter(int sr, int n_fft, int n_mels, int fmin, int fmax, bool htk) {
  int n_f = n_fft / 2 + 1;
  Vectorf fft_freqs = (Vectorf::LinSpaced(n_f, 0.f, static_cast<float>(n_f - 1)) * sr) / n_fft;

  float f_min = 0.f;
  float f_sp = 200.f / 3.f;
  float min_log_hz = 1000.f;
  float min_log_mel = (min_log_hz - f_min) / f_sp;
  float logstep = logf(6.4f) / 27.f;

  auto hz_to_mel = [=](int hz, bool htk) -> float {
    if (htk) {
      return 2595.0f * log10f(1.0f + hz / 700.0f);
    }
    float mel = (hz - f_min) / f_sp;
    if (hz >= min_log_hz) {
      mel = min_log_mel + logf(hz / min_log_hz) / logstep;
    }
    return mel;
  };
  auto mel_to_hz = [=](Vectorf &mels, bool htk) -> Vectorf {
    if (htk) {
      return 700.0f *
             (Vectorf::Constant(n_mels + 2, 10.f).array().pow(mels.array() / 2595.0f) - 1.0f);
    }
    return (mels.array() > min_log_mel)
        .select(((mels.array() - min_log_mel) * logstep).exp() * min_log_hz,
                (mels * f_sp).array() + f_min);
  };

  float min_mel = hz_to_mel(fmin, htk);
  float max_mel = hz_to_mel(fmax, htk);
  Vectorf mels = Vectorf::LinSpaced(n_mels + 2, min_mel, max_mel);
  Vectorf mel_f = mel_to_hz(mels, htk);
  Vectorf fdiff = mel_f.segment(1, mel_f.size() - 1) - mel_f.segment(0, mel_f.size() - 1);
  Matrixf ramps =
      mel_f.replicate(n_f, 1).transpose().array() - fft_freqs.replicate(n_mels + 2, 1).array();

  Matrixf lower = -ramps.topRows(n_mels).array() /
                  fdiff.segment(0, n_mels).transpose().replicate(1, n_f).array();
  Matrixf upper = ramps.bottomRows(n_mels).array() /
                  fdiff.segment(1, n_mels).transpose().replicate(1, n_f).array();
  Matrixf weights = (lower.array() < upper.array()).select(lower, upper).cwiseMax(0);

  auto enorm = (2.0 / (mel_f.segment(2, n_mels) - mel_f.segment(0, n_mels)).array())
                   .transpose()
                   .replicate(1, n_f);
  weights = weights.array() * enorm;

  return weights.transpose();
}

MelFeatureExtract::MelFeatureExtract(int num_frames, int sr, int n_fft, int n_hop, int n_mel,
                                     int fmin, int fmax, const std::string &mode, bool htk,
                                     bool center /*=true*/, int power /*=2*/,
                                     bool is_log /*=true*/) {
  num_fft_ = n_fft;
  num_mel_ = n_mel;
  sample_rate_ = sr;
  num_hop_ = n_hop;
  fmin_ = fmin;
  fmax_ = fmax;
  center_ = center;
  power_ = power;
  mode_ = mode;
  int pad_len = center ? n_fft / 2 : 0;
  pad_len_ = pad_len;

  window_ =
      0.5 *
      (1.f - (Vectorf::LinSpaced(n_fft, 0.f, static_cast<float>(n_fft - 1)) * 2.f * M_PI / n_fft)
                 .array()
                 .cos());
  mel_basis_ = melfilter(sr, n_fft, n_mel, fmin, fmax, htk);
  is_log_ = is_log;
}

void MelFeatureExtract::pad(Vectorf &x, int left, int right, const std::string &mode, float value) {
  // Vectorf x_pad_ = Vectorf::Constant(left+x.size()+right, value);
  if (x_pad_.size() == 0) {
    x_pad_ = Vectorf::Constant(left + right + x.size(), 0);
  }
  x_pad_.segment(left, x.size()) = x;

  if (mode.compare("reflect") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = x[left - i];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + x.size()] = x[x.size() - 2 - i + left];
    }
  }

  if (mode.compare("symmetric") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = x[left - i - 1];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + x.size()] = x[x.size() - 1 - i + left];
    }
  }

  if (mode.compare("edge") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = x[0];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + x.size()] = x[x.size() - 1];
    }
  }
}

static Matrixcf stft(Vectorf &x_paded, Vectorf &window, int n_fft, int n_hop,
                     const std::string &win, bool center, const std::string &mode) {
  // hanning
  // Vectorf window = 0.5*(1.f-(Vectorf::LinSpaced(n_fft, 0.f,
  // static_cast<float>(n_fft-1))*2.f*M_PI/n_fft).array().cos());

  // int pad_len = center ? n_fft / 2 : 0;
  // Vectorf x_paded = pad(x, pad_len, pad_len, mode, 0.f);

  int n_f = n_fft / 2 + 1;
  int n_frames = 1 + (x_paded.size() - n_fft) / n_hop;
  Matrixcf X(n_frames, n_fft);
  Eigen::FFT<float> fft;

  for (int i = 0; i < n_frames; ++i) {
    Vectorf segment = x_paded.segment(i * n_hop, n_fft);
    Vectorf x_frame = window.array() * x_paded.segment(i * n_hop, n_fft).array();
    X.row(i) = fft.fwd(x_frame);
  }
  return X.leftCols(n_f);
}

static Matrixf spectrogram(Matrixcf &X, float power = 1.f) {
  return X.cwiseAbs().array().pow(power);
}

// static Matrixf melspectrogram(Vectorf &x, int sr, int n_fft, int n_hop,
//                         const std::string &win, bool center,
//                         const std::string &mode, float power,
//                         int n_mels, int fmin, int fmax){
//   int pad_len = center ? n_fft / 2 : 0;

//   Matrixcf X = stft(x, n_fft, n_hop, win, center, mode);
//   Matrixf mel_basis = melfilter(sr, n_fft, n_mels, fmin, fmax);
//   Matrixf sp = spectrogram(X, power);
//   Matrixf mel = mel_basis*sp.transpose();
//   return mel;
// }

// static Matrixf power2db(Matrixf& x) {
//   auto log_sp = 10.0f*x.array().max(1e-10).log10();
//   return log_sp.cwiseMax(log_sp.maxCoeff() - 80.0f);
// }
void MelFeatureExtract::update_data(short *p_data, int data_len) {
  if (x_pad_.cols() == 0) {
    x_pad_ = Vectorf::Constant(pad_len_ * 2 + data_len, 0);
  }
  int num_len = int(x_pad_.size()) - 2 * pad_len_;
  if (num_len != data_len) {
    LOGE("size error\n");
  }
  for (int i = 0; i < data_len; i++) {
    x_pad_[i + pad_len_] = p_data[i] / 32768.0;
  }
  int left = pad_len_;
  int right = pad_len_;
  if (mode_.compare("reflect") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 2 - i + left] / 32768.0;
    }
  }

  if (mode_.compare("symmetric") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i - 1] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1 - i + left] / 32768.0;
    }
  }

  if (mode_.compare("edge") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[0] / 32768.0;
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1] / 32768.0;
    }
  }
}
void MelFeatureExtract::update_float_data(float *p_data, int data_len) {
  if (x_pad_.cols() == 0) {
    x_pad_ = Vectorf::Constant(pad_len_ * 2 + data_len, 0);
  }
  int num_len = int(x_pad_.size()) - 2 * pad_len_;
  if (num_len != data_len) {
    LOGE("size error\n");
  }
  for (int i = 0; i < data_len; i++) {
    x_pad_[i + pad_len_] = p_data[i];
  }
  int left = pad_len_;
  int right = pad_len_;
  if (mode_.compare("reflect") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 2 - i + left];
    }
  }

  if (mode_.compare("symmetric") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[left - i - 1];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1 - i + left];
    }
  }

  if (mode_.compare("edge") == 0) {
    for (int i = 0; i < left; ++i) {
      x_pad_[i] = p_data[0];
    }
    for (int i = left; i < left + right; ++i) {
      x_pad_[i + data_len] = p_data[data_len - 1];
    }
  }
}
Matrixf MelFeatureExtract::melspectrogram(std::vector<float> &wav) {
  Vectorf map_x = Eigen::Map<Vectorf>(wav.data(), wav.size());
  int pad_len = center_ ? num_fft_ / 2 : 0;
  pad(map_x, pad_len, pad_len, mode_, 0);
  Matrixcf X = stft(x_pad_, window_, num_fft_, num_hop_, "hann", center_, mode_);
  // Matrixf mel_basis = melfilter(sr, n_fft, n_mels, fmin, fmax);
  Matrixf sp = spectrogram(X, power_);
  Matrixf mel = sp * mel_basis_;

  if (is_log_) {
    for (int r = 0; r < mel.rows(); r++) {
      float *prow = mel.row(r).data();
      for (int c = 0; c < mel.cols(); c++) {
        float v = prow[c];
        if (v < min_val_) v = min_val_;
        prow[c] = 10 * log10f(v);
      }
    }
  }

  return mel;
}
void MelFeatureExtract::melspectrogram_impl(int8_t *p_dst, int dst_len, float q_scale) {
  Matrixcf X = stft(x_pad_, window_, num_fft_, num_hop_, "hann", center_, mode_);
  Matrixf sp = spectrogram(X, power_);
  int matrix_size = sp.rows() * mel_basis_.cols();
  if (matrix_size > dst_len) {
    LOGE("error,dst buffer overflow,expect:%d,got:%d", matrix_size, dst_len);
  }
  // printf("dstsize:%d,%d,buffer_len:%d,qscale:%f\n",sp.rows(),mel_basis_.cols(),dst_len,q_scale);
  // FILE *fp = fopen("/mnt/data/admin1_data/alios_test/logmel.bin","wb");
  for (int r = 0; r < sp.rows(); r++) {
    int8_t *pdst_r = p_dst + r * num_mel_;
    melspec::Vectorf rowv = sp.row(r) * mel_basis_;
    for (int n = 0; n < num_mel_; n++) {
      float v = rowv[n];
      if (v < min_val_) v = min_val_;
      if (is_log_) {
        v = 10 * log10f(v);
      }
      //   rowv[n] = v;
      // }
      // // fwrite(rowv.data(),num_mel_*4,1,fp);
      // for(int n = 0; n < num_mel_;n++){
      //   float v = rowv[n];
      int16_t qval = v * q_scale;
      if (qval < -128) {
        // std::cout<<"overflow qval:"<<qval<<std::endl;
        qval = -128;
      } else if (qval > 127) {
        // std::cout<<"overflow qval:"<<qval<<std::endl;
        qval = 127;
      }
      pdst_r[n] = qval;
    }
    // memcpy(pdst_r,rowv.data(),num_mel_*sizeof(float));
    // Eigen::Map<melspec::Vectorf>(pdst_r, num_mel_) = sp.row(r)*mel_basis_;
    // Vectorf spec = Eigen::Map<Vectorf<float, -1, -1,
    // Eigen::RowMajor>>(p_dst,sp.rows(),mel_basis_.cols());
  }
  // fclose(fp);
  // std::cout<<"finish mel\n";
  // getchar();
  // printf("0x%" PRIXPTR "\n", (uintptr_t)p_dst);
  // std::cout<<"dstaddr:"<<(void*)p_dst<<std::endl;
  // if(is_log_){
  //   std::cout<<"log10 2.3:"<<log10(2.3)<<",dstlen:"<<dst_len;//<<"dstaddr:"<<(void*)p_dst<<std::endl;
  //   getchar();
  //   for(int i = 0; i < dst_len;i++){
  //     float v = p_dst[i];
  //     std::cout<<"data:"<<i<<",val:"<<v<<std::endl;
  //     if(v<min_val_)v=min_val_;
  //     p_dst[i] = 10*log10f(v);
  //   }
  //   std::cout<<"logdone\n";
  // }
}

void MelFeatureExtract::melspectrogram_optimze(short *p_data, int data_len, int8_t *p_dst,
                                               int dst_len, float q_scale) {
  int pad_len = center_ ? num_fft_ / 2 : 0;

  int n_f = num_fft_ / 2 + 1;
  int padded_len = data_len + 2 * pad_len;
  int n_frames = 1 + (padded_len - num_fft_) / num_hop_;

  Eigen::FFT<float> fft;
  melspec::Vectorf segment(num_fft_);
  melspec::Vectorf specmag(n_f);
  const float scale = 1.0 / 32768.0;
  for (int i = 0; i < n_frames; ++i) {
    int start_idx = i * num_hop_;
    for (int j = 0; j < num_fft_; j++) {
      int srcidx = start_idx + j - pad_len;
      if (srcidx < 0) {
        srcidx = -srcidx;
      } else if (srcidx >= data_len) {
        int over = srcidx - data_len;
        srcidx = data_len - over - 2;
      }
      segment[j] = p_data[srcidx] * scale;  // TODO:fuquan.ke this could be optimized
    }
    melspec::Vectorf x_frame = window_.array() * segment.array();
    melspec::Vectorcf spec_ri = fft.fwd(x_frame);
    // compute mag
    melspec::Vectorf specmag = spec_ri.leftCols(n_f).cwiseAbs().array().pow(2);

    int8_t *pdst_r = p_dst + i * num_mel_;
    melspec::Vectorf rowv = specmag * mel_basis_;
    for (int n = 0; n < num_mel_; n++) {
      float v = rowv[n];
      if (v < min_val_) v = min_val_;
      if (is_log_) {
        v = 10 * log10f(v);
      }
      int16_t qval = v * q_scale;
      if (qval < -128) {
        // std::cout<<"overflow qval:"<<qval<<std::endl;
        qval = -128;
      } else if (qval > 127) {
        // std::cout<<"overflow qval:"<<qval<<std::endl;
        qval = 127;
      }
      pdst_r[n] = qval;
    }
  }
}