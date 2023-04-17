#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

namespace cviai {

class IrLiveness final : public Core {
 public:
  IrLiveness();
  virtual ~IrLiveness();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cviai