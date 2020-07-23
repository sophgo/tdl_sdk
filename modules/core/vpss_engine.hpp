#pragma once
#include <cvi_comm_video.h>
#include <cvi_comm_vpss.h>

namespace cviai {

class VpssEngine {
 public:
  VpssEngine();
  ~VpssEngine();
  int init(bool enable_log);
  int stop();
  VPSS_GRP getGrpId();
  int sendFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                const uint32_t enable_chns);
  int sendCropGrpFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int sendCropChnFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout = 100);
  int releaseFrame(VIDEO_FRAME_INFO_S *frame, int chn_idx);

 private:
  void enableLog();

  bool m_enable_log = false;
  bool m_is_vpss_init = false;
  VPSS_GRP m_grpid = -1;
  uint32_t m_enabled_chn = -1;
};
}  // namespace cviai