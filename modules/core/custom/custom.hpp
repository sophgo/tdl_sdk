#pragma once
#include "core.hpp"
#include "core/cviai_custom.h"

namespace cviai {

struct CustomSQParam {
  uint32_t idx = 0;
  bool keep_aspect_ratio = true;
  bool use_model_threashold = true;
  std::vector<float> factor;
  std::vector<float> mean;
};

class Custom final : public Core {
 public:
  explicit Custom();
  int setSQParam(const uint32_t idx, const float *factor, const float *mean, const uint32_t length,
                 const bool use_model_threshold, const bool keep_aspect_ratio);
  int setPreProcessFunc(preProcessFunc func, bool use_tensor_input, bool use_vpss_sq);
  int getInputNum();
  int getInputShape(const char *tensor_name, uint32_t *n, uint32_t *c, uint32_t *h, uint32_t *w);
  int getInputShape(const uint32_t idx, uint32_t *n, uint32_t *c, uint32_t *h, uint32_t *w);
  int inference(VIDEO_FRAME_INFO_S *inFrames, uint32_t num_of_frames);
  int getOutputTensor(const char *tensor_name, int8_t **tensor, uint32_t *tensor_count,
                      uint16_t *unit_size);
  using Core::getInputShape;

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  preProcessFunc preprocessfunc = NULL;

  std::vector<CustomSQParam> m_sq_params;
  std::vector<VIDEO_FRAME_INFO_S> m_processed_frames;
};
}  // namespace cviai