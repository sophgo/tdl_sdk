#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class ImageClassification final : public Core {
 public:
  ImageClassification();
  virtual ~ImageClassification();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_class_meta_t *meta);
  void set_param(VpssPreParam *p_preprocess_cfg);

 private:
  virtual int onModelOpened() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(cvtdl_class_meta_t *meta);
  VpssPreParam *p_preprocess_cfg_;
};
}  // namespace cvitdl
