#pragma once
#include <cvi_comm_vb.h>
#include "cvi_comm.h"

namespace cviai {

class __attribute__((visibility("default"))) VpssEngine {
 public:
  VpssEngine();
  ~VpssEngine();
  int init(VPSS_GRP grp_id = (VPSS_GRP)-1, CVI_U8 device = 0);
  int stop();
  VPSS_GRP getGrpId();
  int sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                const uint32_t enable_chns);
  int sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                const VPSS_SCALE_COEF_E *coeffs, const uint32_t enable_chns);
  int sendCropGrpFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns);
  int sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                       const VPSS_CHN_ATTR_S *chn_attr, const VPSS_SCALE_COEF_E *coeffs,
                       const uint32_t enable_chns);
  int sendCropGrpChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *grp_crop_attr,
                          const VPSS_CROP_INFO_S *chn_crop_attr, const VPSS_CHN_ATTR_S *chn_attr,
                          const uint32_t enable_chns);
  int getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout = 100);
  int releaseFrame(VIDEO_FRAME_INFO_S *frame, int chn_idx);

  void attachVBPool(VB_POOL pool_id);
  VB_POOL getVBPool() const;

 private:
  inline int sendFrameBase(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *grp_crop_attr,
                           const VPSS_CROP_INFO_S *chn_crop_attr, const VPSS_CHN_ATTR_S *chn_attr,
                           const VPSS_SCALE_COEF_E *coeffs, const uint32_t enable_chns);

  bool m_is_vpss_init = false;
  VPSS_GRP m_grpid = -1;
  uint32_t m_enabled_chn = -1;
  VB_POOL m_vbpool_id = VB_INVALID_POOLID;
  uint32_t m_available_max_chn = VPSS_MAX_CHN_NUM;
  CVI_U8 m_dev = 0;
  VPSS_CROP_INFO_S m_crop_attr_reset;
};
}  // namespace cviai
