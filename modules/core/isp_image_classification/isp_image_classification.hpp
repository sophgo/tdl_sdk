#pragma once
#include <bitset>
#include "core/object/cvtdl_object_types.h"
#include "core_internel.hpp"

namespace cvitdl {

class IspImageClassification final : public Core {
 public:
  IspImageClassification();
  virtual ~IspImageClassification();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_class_meta_t *meta, cvtdl_isp_meta_t *isparg);
  void set_param(VpssPreParam *p_preprocess_cfg);
  virtual bool allowExportChannelAttribute() const override { return true; }
  std::vector<int> TopKIndex(std::vector<float> &vec, int topk);

 private:
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;
  int vpssPreprocess(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                     VPSSConfig &vpss_config) override;
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  void outputParser(cvtdl_class_meta_t *meta);
  VpssPreParam *p_preprocess_cfg_;
};
}  // namespace cvitdl
