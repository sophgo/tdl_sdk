#include "motion_detect/motion_detect.hpp"
#if defined(USE_CV181X)
#include "cvi_motion_detect/cvi_motion_detect.hpp"
#elif defined(USE_CV184X_OR_CMODEL_CV184X)
#include "bm_motion_detect/bm_motion_detect.hpp"
#endif
#include "utils/tdl_log.hpp"

MotionDetection::MotionDetection() {}

MotionDetection::~MotionDetection() {}

std::shared_ptr<MotionDetection> MotionDetection::getMotionDetection() {
#if defined(USE_CV184X_OR_CMODEL_CV184X)
  return std::make_shared<BmMotionDetection>();
#elif defined(USE_CV181X)
  return std::make_shared<CviMotionDetection>();
#else
  throw std::invalid_argument("Only support CV184X, CV181X motion detection");
#endif
}
