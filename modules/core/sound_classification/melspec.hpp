
#include <string>
#include <vector>
#include "Eigen/Core"
namespace melspec {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif  // !M_PI

typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;
typedef Eigen::Matrix<std::complex<float>, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorcf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrixf;
typedef Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrixcf;
class MelFeatureExtract {
 public:
  MelFeatureExtract(int num_frames, int sr, int n_fft, int n_hop, int n_mel, int fmin, int fmax,
                    const std::string &mode, bool htk, bool center = true, int power = 2,
                    bool is_log = true);
  ~MelFeatureExtract() {}

  void update_data(short *p_data, int data_len);
  void update_float_data(float *p_data, int data_len);
  void melspectrogram_impl(int8_t *p_dst, int dst_len, float q_scale);
  Matrixf melspectrogram(std::vector<float> &wav);
  void melspectrogram_optimze(short *p_data, int data_len, int8_t *p_dst, int dst_len,
                              float q_scale);
  void pad(Vectorf &x, int left, int right, const std::string &mode, float value);
  // void stft(Vectorf &x, int n_fft, int n_hop, const std::string &win, bool center, const
  // std::string &mode);
 private:
  // float *mp_buffer;
  Matrixf mel_basis_;
  Vectorf x_pad_;
  Vectorf window_;

  int num_fft_;
  int win_len_;
  int num_hop_;
  int num_mel_;
  int sample_rate_;
  int fmin_;
  int fmax_;
  bool center_ = true;
  int power_ = 2;
  std::string mode_;
  int pad_len_ = 0;
  const float min_val_ = 1.0e-6;
  bool is_log_;
};

}  // namespace melspec
