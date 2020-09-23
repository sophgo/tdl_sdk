#pragma once
#include <cvi_comm_video.h>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace cviai {
namespace service {

typedef struct {
  float x;
  float y;
} tracker_pts_t;

class Tracker {
 public:
  int registerId(const VIDEO_FRAME_INFO_S *frame, const int64_t &id, const float x, const float y);
  int getLatestPos(const int64_t &id, float *x, float *y);

 private:
  std::map<int64_t, std::vector<std::pair<CVI_U64, tracker_pts_t>>> m_tracker;
  CVI_U64 m_timestamp;
  CVI_U64 m_deleteduration = 3000;  // 3ms
};
}  // namespace service
}  // namespace cviai
