#pragma once
#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"

namespace cvitdl {

class MaskFaceRecognition final : public Core {
 public:
  MaskFaceRecognition();
  virtual ~MaskFaceRecognition();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *meta);

 private:
  void outputParser(cvtdl_face_t *meta, int meta_i);
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;
  CVI_S32 allocateION();
  void releaseION();

  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cvitdl