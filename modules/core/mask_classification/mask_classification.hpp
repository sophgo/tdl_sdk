#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

namespace cviai {

class MaskClassification final : public Core {
 public:
  MaskClassification();
  virtual ~MaskClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cviai