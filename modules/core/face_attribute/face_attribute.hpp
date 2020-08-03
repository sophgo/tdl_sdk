#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class FaceAttribute final : public Core {
 public:
  FaceAttribute();
  virtual ~FaceAttribute();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_face_t *meta);

 private:
  void prepareInputTensor(cv::Mat src_image, cvai_face_info_t &face_info);
  void outputParser(cvai_face_t *meta, int meta_i);

  float *attribute_buffer = nullptr;
};
}  // namespace cviai