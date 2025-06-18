
#pragma once
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class Blip_Vqa_Tdec final : public Core {
 public:
  Blip_Vqa_Tdec();
  virtual ~Blip_Vqa_Tdec();
  int inference(cvtdl_image_embeds* embeds_meta, cvtdl_tokens* tokens_meta);

 private:
};

}  // namespace cvitdl