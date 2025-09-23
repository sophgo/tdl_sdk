#ifndef FRAME_DUMP_H
#define FRAME_DUMP_H

#include "common/common_types.hpp"
#include "cvi_comm_sys.h"
#include "cvi_comm_vpss.h"
#include "cvi_sys.h"
#include "cvi_vpss.h"

class FrameDump {
 public:
  static int32_t saveFrame(char *filename, VIDEO_FRAME_INFO_S *pstVideoFrame);
};

#endif  // FRAME_DUMP_H