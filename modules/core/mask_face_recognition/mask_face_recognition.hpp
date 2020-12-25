#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class MaskFaceRecognition final : public Core {
 public:
  MaskFaceRecognition();
  virtual ~MaskFaceRecognition();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_face_t *meta);

 private:
  void outputParser(cvai_face_t *meta, int meta_i);
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;

  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai