#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class SuperResolution final : public Core {
 public:
  SuperResolution();
  virtual ~SuperResolution();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_sr_feature *srfeature);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(VIDEO_FRAME_INFO_S *dstFrame);
};
}  // namespace cvitdl
