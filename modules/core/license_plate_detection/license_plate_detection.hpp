#pragma once
#include <cvi_comm_vb.h>
#include "core.hpp"
#include "core/object/cvai_object_types.h"

#include "license_plate_detection_utils.hpp"
#include "opencv2/opencv.hpp"

namespace cviai {

/* WPODNet */
class LicensePlateDetection final : public Core {
 public:
  LicensePlateDetection();
  virtual ~LicensePlateDetection();
  int inference(VIDEO_FRAME_INFO_S *frame, cvai_object_t *vehicle_meta,
                cvai_object_t *license_plate_meta);

 private:
  bool reconstruct(float *t_prob, float *t_trans, CornerPts &c_pts, float threshold_prob = 0.9);
  void prepareInputTensor(cv::Mat &input_mat);

  VB_BLK m_gdc_blk = (VB_BLK)-1;
  VIDEO_FRAME_INFO_S m_wrap_frame;
};
}  // namespace cviai
