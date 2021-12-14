#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

namespace cviai {

class Deeplabv3 final : public Core {
 public:
  Deeplabv3();
  virtual ~Deeplabv3();
  int inference(VIDEO_FRAME_INFO_S *frame, VIDEO_FRAME_INFO_S *out_frame,
                cvai_class_filter_t *filter);
  virtual bool allowExportChannelAttribute() const override { return true; }

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;
  CVI_S32 allocateION();
  void releaseION();
  int outputParser(cvai_class_filter_t *filter);
  VIDEO_FRAME_INFO_S m_label_frame;
};
}  // namespace cviai
