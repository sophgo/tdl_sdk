#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class HandKeypoint final : public Core {
 public:
  HandKeypoint();
  virtual ~HandKeypoint();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_handpose21_meta_ts *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
};
}  // namespace cviai