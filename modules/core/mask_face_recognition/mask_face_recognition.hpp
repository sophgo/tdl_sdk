#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/face/cvai_face_types.h"

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
  virtual int onModelClosed() override;
  CVI_S32 allocateION();
  void releaseION();

  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai