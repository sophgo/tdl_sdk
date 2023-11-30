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

class FaceQuality final : public Core {
 public:
  FaceQuality();
  virtual ~FaceQuality();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *meta, bool *skip);

 private:
  virtual int setupInputPreprocess(std::vector<InputPreprecessSetup> *data) override;
  virtual int onModelOpened() override;
  virtual int onModelClosed() override;
  CVI_S32 allocateION();
  void releaseION();

  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cvitdl
