#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class FaceAttribute final : public Core {
 public:
  explicit FaceAttribute(bool with_attribute);
  virtual ~FaceAttribute();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta, int face_idx = -1);
  void setHardwareGDC(bool use_wrap_hw);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  void outputParser(cvai_face_t *meta, int meta_i);

  bool m_use_wrap_hw;
  const bool m_with_attribute;
  float *attribute_buffer = nullptr;
  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai