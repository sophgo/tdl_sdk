#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class HandClassification final : public Core {
 public:
  HandClassification();
  virtual ~HandClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_object_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cviai