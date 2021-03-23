#include "area_detect.hpp"
#include "core/cviai_types_mem.h"
#include "cviai_log.hpp"

#include <cvi_type.h>
#include <string.h>

namespace cviai {
namespace service {

inline bool checkValid(const float x, const float y) {
  if (x < 0 || y < 0) {
    return false;
  }
  return true;
}

int AreaDetect::setArea(const cvai_pts_t &pts) {
  if (pts.size < 2) {
    LOGE("Registered points must larger than 2.");
    return CVI_FAILURE;
  }
  m_boundaries.clear();
  m_pts.clear();

  if (!checkValid(pts.x[0], pts.y[0])) {
    LOGE("Coordinate cannot be negative ( %3.f, %3.f)", pts.x[0], pts.y[0]);
    return CVI_FAILURE;
  }
  Eigen::Vector2f first_pts(pts.x[0], pts.y[0]);
  Eigen::Vector2f prev_pts = first_pts;
  for (uint32_t i = 1; i < pts.size; i++) {
    if (!checkValid(pts.x[i], pts.y[i])) {
      LOGE("Coordinate cannot be negative ( %3.f, %3.f)", pts.x[0], pts.y[0]);
      return CVI_FAILURE;
    }
    Eigen::Vector2f curr_pts(pts.x[i], pts.y[i]);
    m_boundaries.push_back(Eigen::Hyperplane<float, 2>::Through(prev_pts, curr_pts));
    m_pts.push_back({prev_pts, curr_pts});
    prev_pts = curr_pts;
  }

  // Close loop.
  if (pts.size > 2) {
    if (first_pts != prev_pts) {
      m_boundaries.push_back(Eigen::Hyperplane<float, 2>::Through(prev_pts, first_pts));
      m_pts.push_back({prev_pts, first_pts});
    }
  }
  LOGI("Boundary registered: size %u\n", (uint32_t)m_boundaries.size());
  return CVI_SUCCESS;
}

int AreaDetect::orientation(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r) {
  int val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

  if (val == 0) return 0;    // colinear
  return (val > 0) ? 1 : 2;  // clock or counterclock wise
}

bool AreaDetect::doIntersect(Eigen::Vector2f p1, Eigen::Vector2f q1, Eigen::Vector2f p2,
                             Eigen::Vector2f q2) {
  // Find the four orientations needed for general and
  // special cases
  int o1 = orientation(p1, q1, p2);
  int o2 = orientation(p1, q1, q2);
  int o3 = orientation(p2, q2, p1);
  int o4 = orientation(p2, q2, q1);

  // General case
  if (o1 != o2 && o3 != o4) return true;

  // Special Cases
  // p1, q1 and p2 are colinear and p2 lies on segment p1q1
  if (o1 == 0 && onSegment(p1, p2, q1)) return true;

  // p1, q1 and q2 are colinear and q2 lies on segment p1q1
  if (o2 == 0 && onSegment(p1, q2, q1)) return true;

  // p2, q2 and p1 are colinear and p1 lies on segment p2q2
  if (o3 == 0 && onSegment(p2, p1, q2)) return true;

  // p2, q2 and q1 are colinear and q1 lies on segment p2q2
  if (o4 == 0 && onSegment(p2, q1, q2)) return true;

  return false;  // Doesn't fall in any of the above cases
}

void AreaDetect::run(const uint64_t &timestamp, const uint64_t &unique_id, const float center_pts_x,
                     const float center_pts_y, cvai_area_detect_e *detect) {
  if (unique_id == (uint64_t)-1) {
    *detect = cvai_area_detect_e::UNKNOWN;
    LOGW("Invalid unique_id %" PRIu64 ".\n", unique_id);
    return;
  }
  if (m_boundaries.size() == 1) {
    float x, y;
    bool has_prev = false;
    if (m_tracker.getLatestPos(unique_id, &x, &y) == CVI_SUCCESS) {
      has_prev = true;
    }
    m_tracker.registerId(timestamp, unique_id, center_pts_x, center_pts_y);
    Eigen::Vector2f curr_pts(center_pts_x, center_pts_y);
    auto &line = m_boundaries[0];
    float curr_dis = line.signedDistance(curr_pts);
    if (curr_dis == 0) {
      if (onSegment(m_pts[0].first, curr_pts, m_pts[0].second)) {
        *detect = cvai_area_detect_e::ON_LINE;
        return;
      }
    }
    if (has_prev) {
      Eigen::Vector2f prev_pts(x, y);
      if (doIntersect(m_pts[0].first, m_pts[0].second, prev_pts, curr_pts)) {
        float prev_dis = line.signedDistance(prev_pts);
        if (prev_dis <= 0 && curr_dis > 0) {
          *detect = cvai_area_detect_e::CROSS_LINE_POS;
        } else if (prev_dis >= 0 && curr_dis < 0) {
          *detect = cvai_area_detect_e::CROSS_LINE_NEG;
        } else {
          *detect = cvai_area_detect_e::UNKNOWN;
        }
      } else {
        *detect = cvai_area_detect_e::NO_INTERSECT;
      }
    } else {
      *detect = cvai_area_detect_e::UNKNOWN;
    }
  } else if (m_boundaries.size() >= 3) {
    m_tracker.registerId(timestamp, unique_id, center_pts_x, center_pts_y);

    Eigen::Vector2f prev_pts(-1.f, center_pts_y);
    Eigen::Vector2f curr_pts(center_pts_x, center_pts_y);

    int count = 0;
    for (uint32_t j = 0; j < m_pts.size(); j++) {
      if (doIntersect(prev_pts, curr_pts, m_pts[j].first, m_pts[j].second)) {
        count++;
      }
    }
    *detect = count & 1 ? cvai_area_detect_e::INSIDE_POLYGON : cvai_area_detect_e::OUTSIDE_POLYGON;
  } else {
    *detect = cvai_area_detect_e::UNKNOWN;
    LOGE("Boundary setting error %u\n", (uint32_t)m_boundaries.size());
    return;
  }
  return;
}

bool AreaDetect::onSegment(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r) {
  if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
      q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]))
    return true;
  return false;
}

}  // namespace service
}  // namespace cviai