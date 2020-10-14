#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/cviai_custom.h"
#include "core/object/cvai_object_types.h"

namespace cviai {

class Custom final : public Core {
 public:
  explicit Custom();
  int setSQParam(const float *factor, const float *mean, const uint32_t length,
                 const float threshold, const bool keep_aspect_ratio);
  int setPreProcessFunc(preProcessFunc func, bool use_tensor_input, bool use_vpss_sq);
  int setSkipPostProcess(const bool skip);
  int inference(VIDEO_FRAME_INFO_S *stInFrame);
  int getOutputTensor(const char *tensorName, int8_t *tensor, uint32_t *tensor_count,
                      uint16_t *unit_size);

 private:
  virtual int initAfterModelOpened() override;
  preProcessFunc preprocessfunc = NULL;

  bool m_keep_aspect_ratio = true;
  std::vector<float> m_factor;
  std::vector<float> m_mean;
  float m_quant_threshold;
};
}  // namespace cviai