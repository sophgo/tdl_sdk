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
  virtual int initAfterModelOpened() override;
  void outputParser(int image_width, int image_height, std::vector<cvai_face_info_t> *bboxes_nms);
  void initFaceMeta(cvai_face_t *meta, int size);

  std::vector<cvai_bbox_t> m_all_anchors;
};
}  // namespace cviai