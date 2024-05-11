#pragma once
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class Clip final : public Core {
 public:
  Clip();
  virtual ~Clip();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_clip_feature *clip_feature);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
};
}  // namespace cvitdl