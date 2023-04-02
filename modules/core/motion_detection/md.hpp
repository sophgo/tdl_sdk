#pragma once
#include <memory>
#include "cvi_comm.h"
#include "cvi_comm_vb.h"
#include "ive.hpp"
#include "profiler.hpp"

class MotionDetection {
 public:
  MotionDetection() = delete;
  MotionDetection(ive::IVE *ive_instance);
  ~MotionDetection();

  CVI_S32 init(VIDEO_FRAME_INFO_S *init_frame);
  CVI_S32 detect(VIDEO_FRAME_INFO_S *frame, std::vector<std::vector<float>> &objs,
                 uint8_t threshold, double min_area);
  CVI_S32 update_background(VIDEO_FRAME_INFO_S *frame);
  CVI_S32 get_motion_map(VIDEO_FRAME_INFO_S *frame);
  CVI_S32 set_roi(int x1, int y1, int x2, int y2);
  ive::IVE *get_ive_instance() { return ive_instance; }

 private:
  CVI_S32 construct_images(VIDEO_FRAME_INFO_S *init_frame);
  void free_all();

  CVI_S32 copy_image(VIDEO_FRAME_INFO_S *srcframe, ive::IVEImage *dst);
  ive::IVE *ive_instance;
  void *p_ccl_instance = NULL;
  ive::IVEImage background_img;
  ive::IVEImage md_output;
  ive::IVEImage tmp_cpy_img_;
  ive::IVEImage tmp_src_img_;
  uint32_t im_width;
  uint32_t im_height;
  int m_roi_[4] = {0};
  struct Padding {
    uint32_t left;
    uint32_t top;
    uint32_t right;
    uint32_t bottom;
  };

  Padding m_padding;
  bool use_roi_ = false;
  Timer md_timer_;
};
