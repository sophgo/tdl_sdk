#pragma once
#include "core.hpp"
#include "core/object/cvai_object_types.h"

#include "opencv2/core.hpp"

namespace cviai {

class AlphaPose final : public Core {
 public:
  AlphaPose();
  virtual ~AlphaPose();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_object_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void prepareInputTensor(const cvai_bbox_t &bbox, cv::Mat img_bgr, cvai_bbox_t &align_bbox);
};
}  // namespace cviai