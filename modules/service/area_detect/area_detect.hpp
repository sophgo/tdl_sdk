#pragma once
#include "Eigen/Geometry"
#include "core/core/cvai_core_types.h"
#include "core/cviai_types_mem_internal.h"
#include "service/cviai_service_types.h"
#include "tracker/tracker.hpp"

#include <cvi_comm_video.h>
#include <utility>

namespace cviai {
namespace service {

class AreaDetect {
 public:
  int setArea(const VIDEO_FRAME_INFO_S *frame, const cvai_pts_t &pts);
  int run(const VIDEO_FRAME_INFO_S *frame, const area_detect_t *input, const uint32_t input_length,
          std::vector<cvai_area_detect_e> *status);

 private:
  bool onSegment(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r);
  bool interSegment(Eigen::Hyperplane<float, 2> line, Eigen::Vector2f linepts1,
                    Eigen::Vector2f linepts2, Eigen::Vector2f p, Eigen::Vector2f r);
  std::vector<Eigen::Hyperplane<float, 2>> m_boundaries;
  std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> m_pts;
  Tracker m_tracker;
};
}  // namespace service
}  // namespace cviai