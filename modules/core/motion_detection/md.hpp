#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include "core/object/cvai_object_types.h"
#include "cvi_comm_vb.h"
#include "cvi_common.h"
#include "ive/ive.h"

namespace cviai {
class VpssEngine;
}

class MotionDetection {
 public:
  MotionDetection() = delete;
  MotionDetection(IVE_HANDLE handle, uint32_t th, double _min_area, uint32_t timeout,
                  cviai::VpssEngine *engine);
  ~MotionDetection();

  CVI_S32 init(VIDEO_FRAME_INFO_S *init_frame);
  CVI_S32 detect(VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj_meta);
  CVI_S32 update_background(VIDEO_FRAME_INFO_S *frame);

 private:
  void construct_bbox(std::vector<cv::Rect> bboxes, cvai_object_t *obj_meta);
  CVI_S32 do_vpss_ifneeded(VIDEO_FRAME_INFO_S *srcframe,
                           std::shared_ptr<VIDEO_FRAME_INFO_S> &frame);
  CVI_S32 vpss_process(VIDEO_FRAME_INFO_S *srcframe, VIDEO_FRAME_INFO_S *dstframe);
  CVI_S32 copy_image(VIDEO_FRAME_INFO_S *srcframe, IVE_IMAGE_S *dst);
  IVE_HANDLE ive_handle;
  IVE_SRC_IMAGE_S background_img;
  IVE_IMAGE_S md_output;
  uint32_t count;
  uint32_t threshold;
  double min_area;
  uint32_t im_width;
  uint32_t im_height;
  cviai::VpssEngine *m_vpss_engine;
  uint32_t m_vpss_timeout;
};