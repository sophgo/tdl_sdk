#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvtdl_object_types.h"
#include "decode_tool.hpp"

#include "opencv2/core.hpp"

namespace cvitdl {

/* WPODNet */
class LicensePlateRecognition final : public Core {
 public:
  LicensePlateRecognition(LP_FORMAT format);
  virtual ~LicensePlateRecognition();
  int inference(VIDEO_FRAME_INFO_S *frame, cvtdl_object_t *vehicle_plate_meta);

 private:
  void prepareInputTensor(cv::Mat &input_mat);
  LP_FORMAT format;
  int lp_height, lp_width;
};
}  // namespace cvitdl
