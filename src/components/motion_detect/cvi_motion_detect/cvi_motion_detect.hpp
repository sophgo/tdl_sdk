#pragma once
#include <memory>

#include "motion_detect/motion_detect.hpp"
#include "utils/ive.hpp"
#include "utils/profiler.hpp"

struct Padding {
  uint32_t left;
  uint32_t top;
  uint32_t right;
  uint32_t bottom;
};

class CviMotionDetection : public MotionDetection {
 public:
  CviMotionDetection();
  ~CviMotionDetection();
  int32_t setBackground(
      const std::shared_ptr<BaseImage> &background_image) override;
  int32_t setROI(const std::vector<ObjectBoxInfo> &_roi_s) override;
  int32_t detect(const std::shared_ptr<BaseImage> &image,
                 uint8_t threshold,
                 double min_area,
                 std::vector<std::vector<float>> &objs) override;

 private:
  int32_t constructImages(VIDEO_FRAME_INFO_S *init_frame);

  ive::IVE *ive_instance_;
  void *ccl_instance_;
  ive::IVEImage background_img_;
  ive::IVEImage md_output_;
  ive::IVEImage tmp_cpy_img_;
  ive::IVEImage tmp_src_img_;
  uint32_t im_width_;
  uint32_t im_height_;
  std::vector<ObjectBoxInfo> roi_s_;

  Padding m_padding_;
  bool use_roi_ = false;
  Timer md_timer_;
};
