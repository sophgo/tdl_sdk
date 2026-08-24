/*
 * aeroear_classification.hpp — AeroEar acoustic drone detection model class
 *
 * Encapsulates the full AeroEar inference pipeline (preprocessing + TPU encoder
 * + CPU GRU + detection) behind the standard TDL_Classification API.
 * Inherits BaseModel, following the TDL_OPENMODEL pattern.
 * Input: raw PCM int16 audio (via BaseImage).
 * Output: classification result (background/drone) via ModelClassificationInfo.
 */

#ifndef AEROEAR_CLASSIFICATION_HPP
#define AEROEAR_CLASSIFICATION_HPP

#include "model/base_model.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "aeroear_gru.h"
#include "aeroear_preprocess.h"
#ifdef __cplusplus
}
#endif

#include <vector>

class AeroEarClassification : public BaseModel {
 public:
  AeroEarClassification();
  virtual ~AeroEarClassification();

  int32_t setupNetwork(NetParam &net_param) override;
  int32_t inference(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas,
      const std::map<std::string, float> &parameters = {}) override;
  virtual int32_t outputParse(
      const std::vector<std::shared_ptr<BaseImage>> &images,
      std::vector<std::shared_ptr<ModelOutputInfo>> &out_datas) override;
  virtual int32_t onModelOpened() override;

  /* Get SNR/Activity stats from the last inference for external logging. */
  void getDetectionStats(float &avg_snr, float &avg_act, int &detection_frames,
                         int &detection_segments) const;

 private:
  /* Convert int16 PCM to float32, then compute dual-window log-mel spectrogram.
   */
  void computeLogMel(const int16_t *pcm, size_t n_samples, float *mel_short_tpu,
                     float *mel_long_all);
  /* TPU encoder inference: CNN + proj -> 256-dim features. */
  int32_t encoderInference(const float *mel_short_tpu,
                           const float *mel_long_all, float *proj_out);

  /* Preprocessing context (Hann windows, FFT trig tables). */
  AeroEarContext ctx_;
  bool ctx_inited_ = false;

  /*
   * GRU state persists across inference calls for temporal continuity.
   * First call: init h=0, run all 153 steps.
   * Subsequent calls: keep h, run all 153 steps. The zero-padded front
   * frames flush old drone state; recent-audio tail frames set new state.
   */
  AeroEarGRU gru_;
  bool gru_inited_ = false;

  /* Work buffers */
  std::vector<float> mel_short_all_;
  std::vector<float> mel_long_all_;
  std::vector<float> mel_short_tpu_;
  std::vector<float> proj_out_;
  std::vector<float> snr_out_;
  std::vector<float> act_out_;

  /* Detection stats from last inference */
  float avg_snr_ = 0.0f;
  float avg_act_ = 0.0f;
  int detection_frames_ = 0;
  int detection_segments_ = 0;

  /* Preprocessing parameters */
  int sample_rate_ = 48000;
  int time_len_ = 5; /* 5 seconds */
};

#endif  // AEROEAR_CLASSIFICATION_HPP