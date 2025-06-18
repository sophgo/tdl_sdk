
#pragma once
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class Blip_Itm final : public Core {
 public:
  Blip_Itm();
  virtual ~Blip_Itm();
  int inference(VIDEO_FRAME_INFO_S* frame, cvtdl_tokens* tokens_meta, cvtdl_class_meta_t* cls_meta);

 private:
};

}  // namespace cvitdl