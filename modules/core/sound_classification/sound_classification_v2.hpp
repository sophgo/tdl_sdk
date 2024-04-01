#pragma once
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"
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
  AudioAlgParam get_algparam();
  void set_algparam(AudioAlgParam audio_param);

 private:
  float threshold_;
  melspec::MelFeatureExtract *mp_extractor_ = nullptr;
  int top_num = 500;
  float max_rate = 0.2;
  AudioAlgParam audio_param_;
};
}  // namespace cvitdl
