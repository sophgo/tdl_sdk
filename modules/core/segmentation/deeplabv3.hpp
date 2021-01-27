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

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  int outputParser(cvai_class_filter_t *filter);
  virtual bool allowExportChannelAttribute() const override { return true; }
  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VB_BLK m_gdc_blk_resize = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_label_frame;
  VIDEO_FRAME_INFO_S m_label_frame_resize;
};
}  // namespace cviai
