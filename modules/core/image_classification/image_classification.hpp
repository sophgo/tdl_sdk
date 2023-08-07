#pragma once
#include <bitset>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class ImageClassification final : public Core {
 public:
  ImageClassification();
  virtual ~ImageClassification();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_class_meta_t *meta);
  void set_param(VpssPreParam *p_preprocess_cfg);

 private:
  virtual int onModelOpened() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(cvai_class_meta_t *meta);
  VpssPreParam *p_preprocess_cfg_;
};
}  // namespace cviai
