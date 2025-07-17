#ifndef __IMAGE_PROCESSOR_HPP__
#define __IMAGE_PROCESSOR_HPP__

#include <cmath>
#include <cstdio>
#include "cvi_type.h"
#include "image/base_image.hpp"

class ImageProcessor {
 public:
  virtual ~ImageProcessor() = default;

  virtual int32_t subads(std::shared_ptr<BaseImage> &src1,
                         std::shared_ptr<BaseImage> &src2,
                         std::shared_ptr<BaseImage> &dst) = 0;

  virtual int32_t thresholdProcess(std::shared_ptr<BaseImage> &input,
                                   CVI_U32 threshold_type, CVI_U32 threshold,
                                   CVI_U32 max_value,
                                   std::shared_ptr<BaseImage> &output) = 0;

  virtual int32_t twoWayBlending(std::shared_ptr<BaseImage> &left,
                                 std::shared_ptr<BaseImage> &right,
                                 std::shared_ptr<BaseImage> &wgt,
                                 std::shared_ptr<BaseImage> &output) = 0;

  virtual int32_t erode(std::shared_ptr<BaseImage> &input, CVI_U32 kernal_w,
                        CVI_U32 kernal_h,
                        std::shared_ptr<BaseImage> &output) = 0;
  virtual int32_t dilate(std::shared_ptr<BaseImage> &input, CVI_U32 kernal_w,
                         CVI_U32 kernal_h,
                         std::shared_ptr<BaseImage> &output) = 0;
  // 创建匹配器实例
  static std::shared_ptr<ImageProcessor> getImageProcessor(
      const std::string &tpu_kernel_module_path = "");
};

#endif  // __IMAGE_PROCESSOR_HPP__
