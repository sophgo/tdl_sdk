#pragma once

#include <opencv2/opencv.hpp>

namespace cviai {
namespace service {

class PolygonIntersect {
 public:
  int setArea(const std::vector<cv::Point> &pts);
  int intersectArea(const std::vector<cv::Point> &polygon, float *intersectArea);

 private:
  std::vector<cv::Point> m_target_polygon;
};
}  // namespace service
}  // namespace cviai
