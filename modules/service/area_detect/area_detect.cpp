#include "area_detect.hpp"
#include "core/cviai_types_mem.h"
#include "cviai_log.hpp"

#include <string.h>

namespace cviai {
namespace service {

inline bool checkValid(const float width, const float height, const float x, const float y) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return false;
  }
  return true;
}

int AreaDetect::setArea(const VIDEO_FRAME_INFO_S *frame, const cvai_pts_t &pts) {
  if (pts.size < 2) {
    LOGE("Registered points must larger than 2.");
    return CVI_FAILURE;
  }
  m_boundaries.clear();
  m_pts.clear();

  if (!checkValid(frame->stVFrame.u32Width, frame->stVFrame.u32Height, pts.x[0], pts.y[0])) {
    LOGE("Registered points exceeded frame area. (W, H, x, y) = %u %u %3.f %3.f",
         frame->stVFrame.u32Width, frame->stVFrame.u32Height, pts.x[0], pts.y[0]);
    return CVI_FAILURE;
  }
  Eigen::Vector2f first_pts(pts.x[0], pts.y[0]);
  Eigen::Vector2f prev_pts = first_pts;
  for (uint32_t i = 1; i < pts.size; i++) {
    if (!checkValid(frame->stVFrame.u32Width, frame->stVFrame.u32Height, pts.x[i], pts.y[i])) {
      LOGE("Registered points exceeded frame area. (W, H, x, y) = %u %u %3.f %3.f",
           frame->stVFrame.u32Width, frame->stVFrame.u32Height, pts.x[i], pts.y[i]);
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

int AreaDetect::run(const VIDEO_FRAME_INFO_S *frame, const area_detect_pts_t *input,
                    const uint32_t input_length, std::vector<cvai_area_detect_e> *detected) {
  detected->clear();

  if (m_boundaries.size() == 1) {
    for (uint32_t i = 0; i < input_length; ++i) {
      const float &center_pts_x = input[i].x;
      const float &center_pts_y = input[i].y;
      float x, y;
      bool has_prev = false;
      if (m_tracker.getLatestPos(input[i].unique_id, &x, &y) == CVI_SUCCESS) {
        has_prev = true;
      }
      m_tracker.registerId(frame, input[i].unique_id, center_pts_x, center_pts_y);
      Eigen::Vector2f curr_pts(center_pts_x, center_pts_y);
      auto &line = m_boundaries[0];
      float curr_dis = line.signedDistance(curr_pts);
      if (curr_dis == 0) {
        if (onSegment(m_pts[i].first, curr_pts, m_pts[i].second)) {
          detected->push_back(cvai_area_detect_e::ON_LINE);
          continue;
        }
      }
      if (has_prev) {
        Eigen::Vector2f prev_pts(x, y);
        if (interSegment(line, m_pts[i].first, m_pts[i].second, prev_pts, curr_pts)) {
          float prev_dis = line.signedDistance(prev_pts);
          if (prev_dis <= 0 && curr_dis > 0) {
            detected->push_back(cvai_area_detect_e::CROSS_LINE_POS);
          } else if (prev_dis >= 0 && curr_dis < 0) {
            detected->push_back(cvai_area_detect_e::CROSS_LINE_NEG);
          } else {
            detected->push_back(cvai_area_detect_e::UNKNOWN);
          }
        } else {
          detected->push_back(cvai_area_detect_e::NO_INTERSECT);
        }
      } else {
        detected->push_back(cvai_area_detect_e::UNKNOWN);
      }
    }
  } else if (m_boundaries.size() >= 3) {
    for (uint32_t i = 0; i < input_length; ++i) {
      const float &center_pts_x = input[i].x;
      const float &center_pts_y = input[i].y;
      m_tracker.registerId(frame, input[i].unique_id, center_pts_x, center_pts_y);
      Eigen::Vector2f prev_pts(-1.f, center_pts_y);
      Eigen::Vector2f curr_pts(center_pts_x, center_pts_y);
      int stat = 1;
      int count = 0;
      for (uint32_t j = 0; j < m_boundaries.size(); ++j) {
        auto &line = m_boundaries[j];
        float prev_dis = line.signedDistance(prev_pts);
        float curr_dis = line.signedDistance(curr_pts);
        if (curr_dis == 0) {
          auto &pts = m_pts[j];
          if (onSegment(pts.first, curr_pts, pts.second)) {
            stat = 0;
            break;
          }
          continue;
        } else if (prev_dis * curr_dis >= 0) {
          continue;
        }
        count++;
      }
      if (stat == 0) {
        detected->push_back(cvai_area_detect_e::ON_LINE);
      } else if (count & 1) {
        detected->push_back(cvai_area_detect_e::INSIDE_POLYGON);
      } else {
        detected->push_back(cvai_area_detect_e::OUTSIDE_POLYGON);
      }
    }
  } else {
    return CVI_FAILURE;
  }
  return CVI_SUCCESS;
}

bool AreaDetect::onSegment(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r) {
  if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
      q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]))
    return true;
  return false;
}

bool AreaDetect::interSegment(Eigen::Hyperplane<float, 2> line, Eigen::Vector2f linepts1,
                              Eigen::Vector2f linepts2, Eigen::Vector2f p, Eigen::Vector2f r) {
  auto path = Eigen::Hyperplane<float, 2>::Through(p, r);
  Eigen::Vector2f pts = line.intersection(path);
  return onSegment(linepts1, pts, linepts2);
}

}  // namespace service
}  // namespace cviai