#include "ive/image_processor.hpp"
#include <cstdlib>
#include <cstring>
#include "bm_image_processor/bm_image_processor.hpp"

int32_t ImageProcessor::subads(std::shared_ptr<BaseImage> &src1,
                               std::shared_ptr<BaseImage> &src2,
                               std::shared_ptr<BaseImage> &dst) {
  return 0;
}

int32_t ImageProcessor::thresholdProcess(std::shared_ptr<BaseImage> &input,
                                         std::shared_ptr<BaseImage> &output,
                                         CVI_U32 threshold_type,
                                         CVI_U32 threshold, CVI_U32 max_value) {
  return 0;
}

int32_t ImageProcessor::twoWayBlending(std::shared_ptr<BaseImage> &left,
                                       std::shared_ptr<BaseImage> &right,
                                       std::shared_ptr<BaseImage> &output,
                                       CVI_S32 overlay_lx, CVI_S32 overlay_rx,
                                       CVI_U8 *wgt) {
  return 0;
}

std::shared_ptr<ImageProcessor> ImageProcessor::getImageProcessor(
    std::string image_processor_type) {
  if (image_processor_type == "bm") {
    return std::make_shared<BmImageProcessor>();
  }
  return nullptr;
}
