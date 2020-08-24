#pragma once
#include <cvi_comm_video.h>
#include <cvi_comm_vpss.h>

namespace cviai {

class VpssEngine {
 public:
  VpssEngine();
  ~VpssEngine();
  int init(bool enable_log, VPSS_GRP grp_id = (VPSS_GRP)-1);
  int stop();
  VPSS_GRP getGrpId();
  int sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                const uint32_t enable_chns);
  int sendCropGrpFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int sendCropGrpChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *grp_crop_attr,
                          const VPSS_CROP_INFO_S *chn_crop_attr, const VPSS_CHN_ATTR_S *chn_attr,
                          const uint32_t enable_chns);
  int getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout = 100);
  int releaseFrame(VIDEO_FRAME_INFO_S *frame, int chn_idx);

 private:
  inline int sendFrameBase(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *grp_crop_attr,
                           const VPSS_CROP_INFO_S *chn_crop_attr, const VPSS_CHN_ATTR_S *chn_attr,
                           const uint32_t enable_chns);
  void enableLog();

  bool m_enable_log = false;
  bool m_is_vpss_init = false;
  VPSS_GRP m_grpid = -1;
  uint32_t m_enabled_chn = -1;
};
}  // namespace cviai