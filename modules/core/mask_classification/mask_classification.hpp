#pragma once
#include "core.hpp"
#include "core/face/cvtdl_face_types.h"

namespace cvitdl {

class MaskClassification final : public Core {
 public:
  MaskClassification();
  virtual ~MaskClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_face_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cvitdl