#ifndef FILE_SOUND_CLASSIFICATION_V2_HPP
#define FILE_SOUND_CLASSIFICATION_V2_HPP
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "melspec.hpp"
#define SCALE_FACTOR_FOR_INT16 32768.0

namespace cvitdl {

class SoundClassificationV2 final : public Core {
 public:
  SoundClassificationV2();
  virtual ~SoundClassificationV2();
  int onModelOpened();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, int *index);
  int setThreshold(const float th) {
    threshold_ = th;
    return CVI_SUCCESS;
  };
  int getClassesNum();
  int get_top_k(float *result, size_t count);
  void normal_sound(short *temp_buffer, int n);

 private:
  float threshold_;
  int win_len_ = 1024;
  int num_fft_ = 1024;
  int hop_len_ = 256;
  int sample_rate_ = 16000;
  int time_len_ = 3;  // 3 second
  int num_mel_ = 40;
  int fmin_ = 0;
  int fmax_ = sample_rate_ / 2;
  melspec::MelFeatureExtract *mp_extractor_ = nullptr;
  int top_num = 500;
  float max_rate = 0.2;
};
}  // namespace cvitdl
#endif
