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
                                         CVI_U32 threshold_type,
                                         CVI_U32 threshold, CVI_U32 max_value,
                                         std::shared_ptr<BaseImage> &output) {
  return 0;
}

int32_t ImageProcessor::twoWayBlending(std::shared_ptr<BaseImage> &left,
                                       std::shared_ptr<BaseImage> &right,
                                       std::shared_ptr<BaseImage> &wgt,
                                       std::shared_ptr<BaseImage> &output) {
  return 0;
}

std::shared_ptr<ImageProcessor> ImageProcessor::getImageProcessor() {
#if defined(__CV184X__) || defined(__CMODEL_CV184X__)
  return std::make_shared<BmImageProcessor>();
#else
  return nullptr;
#endif
}
