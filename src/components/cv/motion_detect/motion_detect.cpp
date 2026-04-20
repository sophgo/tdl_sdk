#include "cv/motion_detect/motion_detect.hpp"
#if defined(__CV181X__) || defined(__CV186X__)
#include "cvi_motion_detect/cvi_motion_detect.hpp"
#elif defined(__CV184X__) || defined(__CMODEL_CV184X__)
#include "bm_motion_detect/bm_motion_detect.hpp"
#endif
#include "utils/tdl_log.hpp"

MotionDetection::MotionDetection() {}

MotionDetection::~MotionDetection() {}

std::shared_ptr<MotionDetection> MotionDetection::getMotionDetection() {
#if defined(__CV184X__) || defined(__CMODEL_CV184X__)
  return std::make_shared<BmMotionDetection>();
#elif defined(__CV181X__) || defined(__CV186X__)
  return std::make_shared<CviMotionDetection>();
#else
  throw std::invalid_argument(
      "Only support SOPHON, CV184X and CV181X motion detection");
#endif
}
