#pragma once
#include <bitset>

#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class RawImageClassification final : public Core {
 public:
  RawImageClassification();
  virtual ~RawImageClassification();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_class_meta_t *meta);

  std::vector<int> TopKIndex(std::vector<float> &vec, int topk);

 private:
  virtual int onModelOpened() override;

  void outputParser(cvtdl_class_meta_t *meta);
};
}  // namespace cvitdl
