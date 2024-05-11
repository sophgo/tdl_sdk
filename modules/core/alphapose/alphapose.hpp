#pragma once
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"
#include "opencv2/core.hpp"

namespace cvitdl {

class AlphaPose final : public Core {
 public:
  AlphaPose();
  virtual ~AlphaPose();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_object_t *meta);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void prepareInputTensor(const cvtdl_bbox_t &bbox, cv::Mat img_bgr, cvtdl_bbox_t &align_bbox);
};
}  // namespace cvitdl
