#ifndef _CVI_TRACKER_HPP_
#define _CVI_TRACKER_HPP_

#include "cvi_deepsort_types_internal.hpp"

class Tracker {
 public:
  uint64_t id;
  BBOX bbox; /* format: top-left(x, y), width, height */
};

#endif /* _CVI_TRACKER_HPP_ */