#pragma once
#include "core.hpp"
#include "core/face/cvai_face_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

class ThermalFace final : public Core {
 public:
  ThermalFace();
  virtual ~ThermalFace();
  int inference(VIDEO_FRAME_INFO_S *srcFrame, cvai_face_t *meta);

 private:
  virtual int initAfterModelOpened(std::vector<initSetup> *data) override;
  virtual int vpssPreprocess(const std::vector<VIDEO_FRAME_INFO_S *> &srcFrames,
                             std::vector<std::shared_ptr<VIDEO_FRAME_INFO_S>> *dstFrames) override;
  void outputParser(const int image_width, const int image_height, const int frame_width,
                    const int frame_height, cvai_face_t *meta);

  std::vector<cvai_bbox_t> m_all_anchors;
};
}  // namespace cviai