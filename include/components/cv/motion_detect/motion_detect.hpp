#pragma once
#include <memory>

#include "common/model_output_types.hpp"
#include "image/base_image.hpp"

class MotionDetection {
 public:
  MotionDetection();
  ~MotionDetection();
  // 设置背景图像
  virtual int32_t setBackground(
      const std::shared_ptr<BaseImage> &background_image) = 0;
  // 设置ROI
  virtual int32_t setROI(const std::vector<ObjectBoxInfo> &_roi_s) = 0;
  // 检测运动
  virtual int32_t detect(const std::shared_ptr<BaseImage> &image,
                         uint8_t threshold, double min_area,
                         std::vector<ObjectBoxInfo> &objs) = 0;
  // 是否需要设置ROI
  virtual bool isROIEmpty() = 0;
  // 获取运动检测实例
  static std::shared_ptr<MotionDetection> getMotionDetection();
};