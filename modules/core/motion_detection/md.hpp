#pragma once
#include <opencv2/opencv.hpp>
#include "core/object/cvai_object_types.h"
#include "cvi_comm_vb.h"
#include "cvi_common.h"
#include "ive/ive.h"

class MotionDetection {
 public:
  MotionDetection() = delete;
  MotionDetection(IVE_HANDLE handle, VIDEO_FRAME_INFO_S *init_frame, uint32_t th, double _min_area);
  ~MotionDetection();

  CVI_S32 detect(VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj_meta);
  CVI_S32 update_background(VIDEO_FRAME_INFO_S *frame);

 private:
  void construct_bbox(std::vector<cv::Rect> bboxes, cvai_object_t *obj_meta);

  IVE_HANDLE ive_handle;
  IVE_SRC_IMAGE_S src[2], tmp;
  IVE_IMAGE_S bk_dst;
  uint32_t count;
  uint32_t threshold;
  double min_area;
  uint32_t im_width;
  uint32_t im_height;
};