#include "tracker.hpp"
#include "core/cviai_types_mem.h"
#include "core/cviai_types_mem_internal.h"

#include <string.h>
#include <syslog.h>

#define DEFAULT_DT_ZOOM_TRANS_RATIO 0.1f
#include <iostream>
namespace cviai {
namespace service {
int Tracker::registerId(const VIDEO_FRAME_INFO_S *frame, const int64_t &id, const float x,
                        const float y) {
  m_timestamp = frame->stVFrame.u64PTS;
  cvai_pts_t pts;
  memset(&pts, 0, sizeof(cvai_pts_t));
  CVI_AI_MemAlloc(1, &pts);
  pts.x[0] = x;
  pts.y[0] = y;
  m_tracker[id].push_back({m_timestamp, pts});

  auto it = m_tracker.begin();
  while (it != m_tracker.end()) {
    auto &vec = it->second;
    if ((m_timestamp - vec[vec.size() - 1].first) > m_deleteduration) {
      for (uint32_t i = 0; i < vec.size(); ++i) {
        CVI_AI_FreeCpp(&vec[i].second);
      }
      it = m_tracker.erase(it);
    } else {
      ++it;
    }
  }
  return CVI_SUCCESS;
}
int Tracker::getLatestPos(const int64_t &id, float *x, float *y) {
  auto it = m_tracker.find(id);
  if (it != m_tracker.end()) {
    auto &vec = it->second;
    *x = vec[vec.size() - 1].second.x[0];
    *y = vec[vec.size() - 1].second.y[0];
    return CVI_SUCCESS;
  }
  return CVI_FAILURE;
}
}  // namespace service
}  // namespace cviai