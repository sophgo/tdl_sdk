#ifndef __POSE_HELPER_HPP__
#define __POSE_HELPER_HPP__

#include <cstdint>
#include <vector>

typedef struct {
  float yaw;    // 偏航角,degree
  float pitch;  // 俯仰角,degree
  float roll;   // 翻滚角,degree

  float facialUnitNormalVector[3];
} FacePose;

class PoseHelper {
 public:
  static int32_t predictFacePose(const std::vector<float>& landmark_5x,
                                 const std::vector<float>& landmark_5y,
                                 const int img_width, const int img_height,
                                 FacePose* hp);

  static int32_t predictFacePose(const std::vector<float>& landmark_5x,
                                 const std::vector<float>& landmark_5y,
                                 FacePose* hp);
};

#endif  // __POSE_HELPER_HPP__
