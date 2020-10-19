#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class OSNet final : public Core {
 public:
  explicit OSNet();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_object_t *meta, int obj_idx);

 private:
  virtual int initAfterModelOpened(float *factor, float *mean, bool &pad_reverse,
                                   bool &keep_aspect_ratio, bool &use_model_threshold) override;
};
}  // namespace cviai