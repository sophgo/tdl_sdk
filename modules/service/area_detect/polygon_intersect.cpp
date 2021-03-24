#include "polygon_intersect.hpp"
#include <cvi_type.h>
#include "cviai_log.hpp"

namespace cviai {
namespace service {

int PolygonIntersect::setArea(const std::vector<cv::Point> &pts) {
  if (!cv::isContourConvex(pts)) {
    LOGE("polygon should be convex!\n");
    return CVI_FAILURE;
  }

  m_target_polygon.clear();
  m_target_polygon = pts;
  return CVI_SUCCESS;
}

int PolygonIntersect::intersectArea(const std::vector<cv::Point> &polygon, float *intersectArea) {
  if (!cv::isContourConvex(polygon)) {
    LOGE("polygon should be convex!\n");
    *intersectArea = 0;
    return CVI_FAILURE;
  }

  if (m_target_polygon.size() < 3) {
    LOGE("should set target polygon first!\n");
    *intersectArea = 0;
    return CVI_FAILURE;
  }

  std::vector<cv::Point> intersectionPolygon;
  *intersectArea = cv::intersectConvexConvex(m_target_polygon, polygon, intersectionPolygon, false);
  return CVI_SUCCESS;
}

}  // namespace service
}  // namespace cviai