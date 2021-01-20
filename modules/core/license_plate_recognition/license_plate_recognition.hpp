#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

#include "opencv2/opencv.hpp"

namespace cviai {

/* WPODNet */
class LicensePlateRecognition final : public Core {
 public:
  LicensePlateRecognition();
  virtual ~LicensePlateRecognition();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *license_plate_meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);

  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai
