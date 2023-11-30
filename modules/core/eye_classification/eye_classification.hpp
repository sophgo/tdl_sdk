#pragma once
#include <cvi_comm_vb.h>
#include <cvi_sys.h>
#ifdef ATHENA2
#include "core_a2.hpp"
#else
#include "core.hpp"
#endif
#include "core/face/cvtdl_face_types.h"
#include "opencv2/core.hpp"

namespace cvitdl {

class EyeClassification final : public Core {
 public:
  EyeClassification();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_face_t *meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);
};
}  // namespace cvitdl
