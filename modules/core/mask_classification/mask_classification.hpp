#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class MaskClassification final : public Core {
 public:
  MaskClassification();
  virtual ~MaskClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta);

 private:
  virtual int initAfterModelOpened(float *factor, float *mean, bool &pad_reverse,
                                   bool &keep_aspect_ratio, bool &use_model_threshold) override;
};
}  // namespace cviai