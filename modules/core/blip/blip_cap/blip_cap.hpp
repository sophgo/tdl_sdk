
#pragma once
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class Blip_Cap final : public Core {
 public:
  Blip_Cap();
  virtual ~Blip_Cap();
  int inference(VIDEO_FRAME_INFO_S* frame, cvtdl_tokens* tokens_meta);

 private:
};

}  // namespace cvitdl