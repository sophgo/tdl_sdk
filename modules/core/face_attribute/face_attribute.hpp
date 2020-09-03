#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class FaceAttribute final : public Core {
 public:
  FaceAttribute(bool use_wrap_hw);
  virtual ~FaceAttribute();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta, int face_idx);
  void setWithAttribute(bool with_attr);

 private:
  virtual int initAfterModelOpened() override;
  void outputParser(cvai_face_t *meta, int meta_i);

  const bool m_use_wrap_hw;
  bool m_with_attribute = true;
  float *attribute_buffer = nullptr;
  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai