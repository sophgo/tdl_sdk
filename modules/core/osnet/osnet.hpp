#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class OSNet final : public Core {
 public:
  explicit OSNet();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_object_t *meta, int obj_idx);

 private:
  virtual int initAfterModelOpened() override;
};
}  // namespace cviai