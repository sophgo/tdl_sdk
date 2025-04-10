
#pragma once
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class Blip_Vqa_Tenc final : public Core {
 public:
  Blip_Vqa_Tenc();
  virtual ~Blip_Vqa_Tenc();
  int inference(cvtdl_image_embeds* embeds_meta, cvtdl_tokens* tokens_meta);

 private:
};

}  // namespace cvitdl