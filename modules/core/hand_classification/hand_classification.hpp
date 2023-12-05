#pragma once
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class HandClassification final : public Core {
 public:
  HandClassification();
  virtual ~HandClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvtdl_object_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cvitdl