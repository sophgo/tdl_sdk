#pragma once
#include "Eigen/Geometry"
#include "core/core/cvai_core_types.h"
#include "core/cviai_types_mem_internal.h"
#include "service/cviai_service_types.h"
#include "tracker/tracker.hpp"

#include <utility>

namespace cviai {
namespace service {

class AreaDetect {
 public:
  int setArea(const cvai_pts_t &pts);
  void run(const uint64_t &timestamp, const uint64_t &unique_id, const float center_pts_x,
           const float center_pts_y, cvai_area_detect_e *detect);

 private:
  bool onSegment(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r);
  int orientation(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r);
  bool doIntersect(Eigen::Vector2f p1, Eigen::Vector2f q1, Eigen::Vector2f p2, Eigen::Vector2f q2);
  std::vector<Eigen::Hyperplane<float, 2>> m_boundaries;
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> m_pts;
  Tracker m_tracker;
};
}  // namespace service
}  // namespace cviai
