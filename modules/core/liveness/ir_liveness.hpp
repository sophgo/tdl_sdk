#pragma once
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"

namespace cvitdl {

class IrLiveness final : public Core {
 public:
  IrLiveness();
  virtual ~IrLiveness();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_face_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cvitdl