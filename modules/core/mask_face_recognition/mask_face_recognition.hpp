#pragma once
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
  void prepareInputTensor(const cv::Mat &src_image, cvai_face_info_t &face_info);
  void outputParser(cvai_face_t *meta, int meta_i);
};
}  // namespace cviai