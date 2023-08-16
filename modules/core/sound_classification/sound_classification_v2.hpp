#ifndef FILE_SOUND_CLASSIFICATION_V2_HPP
#define FILE_SOUND_CLASSIFICATION_V2_HPP
#include "core.hpp"
#include "melspec.hpp"

namespace cviai {

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
};
}  // namespace cviai
#endif
