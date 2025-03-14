
#pragma once
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class Blip_Vqa_Venc final : public Core {
 public:
  Blip_Vqa_Venc();
  virtual ~Blip_Vqa_Venc();
  int inference(VIDEO_FRAME_INFO_S* frame, cvtdl_image_embeds* embeds_meta);

 private:
};

}  // namespace cvitdl