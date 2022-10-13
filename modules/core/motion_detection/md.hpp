#pragma once
#include <memory>
#include "core/object/cvai_object_types.h"
#include "cvi_comm.h"
#include "cvi_comm_vb.h"
#include "ive.hpp"
#include "profiler.hpp"
namespace cviai {
class VpssEngine;
}

class MotionDetection {
 public:
  MotionDetection() = delete;
  MotionDetection(ive::IVE *ive_instance, uint32_t timeout, cviai::VpssEngine *engine);
  ~MotionDetection();

  CVI_S32 init(VIDEO_FRAME_INFO_S *init_frame);
  CVI_S32 detect(VIDEO_FRAME_INFO_S *frame, cvai_object_t *obj_meta, uint8_t threshold,
                 double min_area);
  CVI_S32 update_background(VIDEO_FRAME_INFO_S *frame);

 private:
  CVI_S32 construct_images(VIDEO_FRAME_INFO_S *init_frame);
  void free_all();

  CVI_S32 do_vpss_ifneeded(VIDEO_FRAME_INFO_S *srcframe,
                           std::shared_ptr<VIDEO_FRAME_INFO_S> &frame);
  CVI_S32 vpss_process(VIDEO_FRAME_INFO_S *srcframe, VIDEO_FRAME_INFO_S *dstframe);
  CVI_S32 copy_image(VIDEO_FRAME_INFO_S *srcframe, ive::IVEImage *dst);
  ive::IVE *ive_instance;
  void *p_ccl_instance = NULL;
  ive::IVEImage background_img;
  ive::IVEImage md_output;
  ive::IVEImage tmp_cpy_img_;
  ive::IVEImage tmp_src_img_;
  uint32_t im_width;
  uint32_t im_height;

  struct Padding {
    uint32_t left;
    uint32_t top;
    uint32_t right;
    uint32_t bottom;
  };

  Padding m_padding;
  cviai::VpssEngine *m_vpss_engine;
  uint32_t m_vpss_timeout;

  Timer md_timer_;
};
