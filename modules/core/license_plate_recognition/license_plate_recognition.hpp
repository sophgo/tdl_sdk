#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"
#include "decode_tool.hpp"

#include "opencv2/opencv.hpp"

namespace cviai {

/* WPODNet */
class LicensePlateRecognition final : public Core {
 public:
  LicensePlateRecognition(LP_FORMAT format);
  virtual ~LicensePlateRecognition();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_plate_meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);
  LP_FORMAT format;
  int lp_height, lp_width;
};
}  // namespace cviai
