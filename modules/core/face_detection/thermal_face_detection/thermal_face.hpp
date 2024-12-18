#pragma once
#include "core/face/cvtdl_face_types.h"
#include "face_detection.hpp"

namespace cvitdl {

class ThermalFace final : public FaceDetectionBase {
 public:
  ThermalFace();
  ~ThermalFace(){};
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvtdl_face_t *meta) override;

 private:
  int onModelOpened() override;

  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvtdl_face_t *meta);

  std::vector<cvtdl_bbox_t> m_all_anchors;
};
}  // namespace cvitdl