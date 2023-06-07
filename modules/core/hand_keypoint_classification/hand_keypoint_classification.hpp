#ifndef FILE_HAND_KEYPOINT_CLASSIFICATION_HPP
#define FILE_HAND_KEYPOINT_CLASSIFICATION_HPP
#include <vector>
#include "Eigen/Core"
#include "core.hpp"
#include "core/object/cvai_object_types.h"
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Vectorf;

namespace cviai {

class HandKeypointClassification final : public Core {
 public:
  HandKeypointClassification();
  virtual ~HandKeypointClassification();
  int inference(VIDEO_FRAME_INFO_S *stOutFrame, cvai_handpose21_meta_t *handpose);

 private:
  Vectorf _data;
};
}  // namespace cviai
#endif