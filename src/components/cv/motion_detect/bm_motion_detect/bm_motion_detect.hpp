#pragma once
#include <memory>

#include "common/model_output_types.hpp"
#include "cv/motion_detect/motion_detect.hpp"
#include "image/base_image.hpp"
#include "ive/image_processor.hpp"
#include "utils/profiler.hpp"

class BmMotionDetection : public MotionDetection {
 public:
  BmMotionDetection();
  ~BmMotionDetection();

  int32_t setBackground(
      const std::shared_ptr<BaseImage> &background_image) override;
  int32_t setROI(const std::vector<ObjectBoxInfo> &_roi_s) override;
  int32_t detect(const std::shared_ptr<BaseImage> &image, uint8_t threshold,
                 double min_area,
                 std::vector<std::vector<float>> &objs) override;
  bool isROIEmpty() override;

 private:
  std::shared_ptr<BaseImage> background_image_;
  std::shared_ptr<BaseImage> md_output_;  // 保存运动检测输出图像
  std::vector<ObjectBoxInfo> roi_s_;
  void *ccl_instance_;
  bool use_roi_ = false;
  std::shared_ptr<ImageProcessor> image_processor_;  // 图像处理器
  Timer md_timer_;                                   // 用于计时
  uint32_t im_width_;                                // 图像宽度
  uint32_t im_height_;                               // 图像高度
};
