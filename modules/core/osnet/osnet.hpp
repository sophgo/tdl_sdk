#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class OSNet final : public Core {
 public:
  explicit OSNet();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_object_t *meta, int obj_idx);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cviai