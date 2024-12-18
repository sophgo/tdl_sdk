#pragma once
#include <bitset>

#include "core.hpp"
#include "core/object/cvtdl_object_types.h"

namespace cvitdl {

class IspImageClassification final : public Core {
 public:
  IspImageClassification();
  virtual ~IspImageClassification();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_class_meta_t *meta, cvtdl_isp_meta_t *isparg);
  std::vector<int> TopKIndex(std::vector<float> &vec, int topk);

 private:
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;

  void outputParser(cvtdl_class_meta_t *meta);
};
}  // namespace cvitdl
