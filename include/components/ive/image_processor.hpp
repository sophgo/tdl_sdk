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
                                   std::shared_ptr<BaseImage> &output,
                                   CVI_U32 threshold_type,
                                   CVI_U32 threshold,
                                   CVI_U32 max_value) = 0;

  virtual int32_t twoWayBlending(std::shared_ptr<BaseImage> &left,
                                 std::shared_ptr<BaseImage> &right,
                                 std::shared_ptr<BaseImage> &output,
                                 CVI_S32 overlay_lx,
                                 CVI_S32 overlay_rx,
                                 CVI_U8 *wgt) = 0;
  // 创建匹配器实例
  static std::shared_ptr<ImageProcessor> getImageProcessor(
      std::string image_processor_type);
};

#endif  // __IMAGE_PROCESSOR_HPP__
