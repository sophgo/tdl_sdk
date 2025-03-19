#ifndef AUDIO_CLASSIFICATION_HPP
#define AUDIO_CLASSIFICATION_HPP
#include "audio_classification/melspec.hpp"
#include "model/base_model.hpp"

class AudioClassification : public BaseModel {
 public:
  AudioClassification();
  AudioClassification(std::pair<int, int> sound_pair);
  virtual ~AudioClassification();
  int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
      const std::map<std::string, float> &parameters = {}) override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

  int32_t getTopK(float *result, size_t count, float* score);
  void normalizeSound(short *temp_buffer, int n);
  int32_t setParameters(
      const std::map<std::string, float> &parameters) override;
  int32_t getParameters(std::map<std::string, float> &parameters) override;
  //   cvitdl_sound_param get_algparam();
  //   void set_algparam(cvitdl_sound_param audio_param);

 private:
  float threshold_;
  melspec::MelFeatureExtract *mp_extractor_ = nullptr;
  int top_num = 500;
  float max_rate = 0.2;
  int win_len_;
  int num_fft_;
  int hop_len_;
  int sample_rate_;
  int time_len_;
  int num_mel_;
  int fmin_;
  int fmax_;
  int fix_;
};

#endif
