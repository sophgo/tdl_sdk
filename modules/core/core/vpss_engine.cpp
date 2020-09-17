#include "vpss_engine.hpp"

#include "core/utils/vpss_helper.h"
#include "cviruntime.h"

#include <syslog.h>
namespace cviai {

VpssEngine::VpssEngine() {}

VpssEngine::~VpssEngine() {}

void VpssEngine::enableLog() {
  // Tunr on Vpss Log
  LOG_LEVEL_CONF_S log_conf;
  log_conf.enModId = (MOD_ID_E)6;  // vpss
  CVI_LOG_GetLevelConf(&log_conf);
  syslog(LOG_INFO, "Set Vpss Log Level: %d, log will save into cvi_mmf_aisdk.log\n",
         log_conf.s32Level);
  log_conf.s32Level = 7;
  CVI_LOG_SetLevelConf(&log_conf);

  CVI_LOG_EnableLog2File(CVI_TRUE, (char *)"cvi_mmf_aisdk.log");
  m_enable_log = true;
}

int VpssEngine::init(bool enable_log, VPSS_GRP grp_id) {
  if (m_is_vpss_init) {
    syslog(LOG_WARNING, "Vpss already init.\n");
    return CVI_FAILURE;
  }
  if (enable_log) {
    enableLog();
  }
  if (CVI_SYS_GetVPSSMode() == VPSS_MODE_DUAL) {
    // FIXME: Currently hardcoded due to no define in mmf.
    m_available_max_chn = VPSS_MAX_CHN_NUM - 1;
  }

  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_CHN_ATTR_S vpss_chn_attr;
  // Not magic number, only for init.
  uint32_t width = 100;
  uint32_t height = 100;
  m_enabled_chn = 1;
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, width, height, PIXEL_FORMAT_YUV_PLANAR_420);
  VPSS_CHN_DEFAULT_HELPER(&vpss_chn_attr, width, height, PIXEL_FORMAT_RGB_888_PLANAR, true);

  /*start vpss*/
  m_grpid = -1;
  if (grp_id != (CVI_U32)-1) {
    if (CVI_VPSS_CreateGrp(grp_id, &vpss_grp_attr) != CVI_SUCCESS) {
      syslog(LOG_ERR, "User assign group id %u failed to create vpss instance.\n", grp_id);
      return CVI_FAILURE;
    }
    m_grpid = grp_id;
  } else {
    int id = CVI_VPSS_GetAvailableGrp();
    if (CVI_VPSS_CreateGrp(id, &vpss_grp_attr) != CVI_SUCCESS) {
      syslog(LOG_ERR, "User assign group id %u failed to create vpss instance.\n", grp_id);
      return CVI_FAILURE;
    }
    m_grpid = id;
  }
  if (m_grpid == (CVI_U32)-1) {
    syslog(LOG_ERR, "All vpss grp init failed!\n");
    return CVI_FAILURE;
  }
  int s32Ret = CVI_VPSS_ResetGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "CVI_VPSS_ResetGrp(grp:%d) failed with %#x!\n", m_grpid, s32Ret);
    return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < m_enabled_chn; i++) {
    s32Ret = CVI_VPSS_SetChnAttr(m_grpid, i, &vpss_chn_attr);

    if (s32Ret != CVI_SUCCESS) {
      syslog(LOG_ERR, "CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
      return CVI_FAILURE;
    }

    s32Ret = CVI_VPSS_EnableChn(m_grpid, i);

    if (s32Ret != CVI_SUCCESS) {
      syslog(LOG_ERR, "CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
      return CVI_FAILURE;
    }
  }
  s32Ret = CVI_VPSS_StartGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return CVI_FAILURE;
  }

  m_is_vpss_init = true;
  return CVI_SUCCESS;
}

int VpssEngine::stop() {
  if (!m_is_vpss_init) {
    syslog(LOG_ERR, "Vpss is not init yet.\n");
    return CVI_FAILURE;
  }

  for (uint32_t j = 0; j < m_enabled_chn; j++) {
    int s32Ret = CVI_VPSS_DisableChn(m_grpid, j);
    if (s32Ret != CVI_SUCCESS) {
      syslog(LOG_ERR, "CVI_VPSS_DisableChn failed with %#x!\n", s32Ret);
      return CVI_FAILURE;
    }
  }

  int s32Ret = CVI_VPSS_StopGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "CVI_VPSS_StopGrp failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  s32Ret = CVI_VPSS_DestroyGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "CVI_VPSS_DestroyGrp failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  if (m_enable_log) {
    CVI_LOG_EnableLog2File(CVI_FALSE, CVI_NULL);
  }
  m_is_vpss_init = false;
  return CVI_SUCCESS;
}

VPSS_GRP VpssEngine::getGrpId() { return m_grpid; }

int VpssEngine::setResizeMethod(const VPSS_CHN chn, const VPSS_SCALE_COEF_E coef) {
  return CVI_VPSS_SetChnScaleCoefLevel(m_grpid, chn, coef);
}

int VpssEngine::getResizeMethod(const VPSS_CHN chn, VPSS_SCALE_COEF_E *coef) {
  return CVI_VPSS_GetChnScaleCoefLevel(m_grpid, chn, coef);
}

int VpssEngine::sendFrameBase(const VIDEO_FRAME_INFO_S *frame,
                              const VPSS_CROP_INFO_S *grp_crop_attr,
                              const VPSS_CROP_INFO_S *chn_crop_attr,
                              const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  if (enable_chns >= m_enabled_chn) {
    for (uint32_t i = m_enabled_chn; i < enable_chns; i++) {
      CVI_VPSS_EnableChn(m_grpid, i);
    }
  } else {
    for (uint32_t i = enable_chns; i < m_enabled_chn; i++) {
      CVI_VPSS_DisableChn(m_grpid, i);
    }
  }
  m_enabled_chn = enable_chns;

  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                          frame->stVFrame.enPixelFormat);
  // Auto choose 1 if more than one channel.
  // Will auto changed to 0 if is SINGLE_MODE when set attr.
  if (m_enabled_chn > 1) {
    vpss_grp_attr.u8VpssDev = 1;
    if (m_enabled_chn > m_available_max_chn) {
      syslog(LOG_ERR, "Exceed max available channel %u. Current: %u.\n", m_available_max_chn,
             m_enabled_chn);
      return CVI_FAILURE;
    }
  }
  int ret = CVI_VPSS_SetGrpAttr(m_grpid, &vpss_grp_attr);
  if (ret != CVI_SUCCESS) {
    syslog(LOG_ERR, "CVI_VPSS_SetGrpAttr failed with %#x\n", ret);
    return ret;
  }
  if (grp_crop_attr != NULL) {
    int ret = CVI_VPSS_SetGrpCrop(m_grpid, grp_crop_attr);
    if (ret != CVI_SUCCESS) {
      syslog(LOG_ERR, "CVI_VPSS_SetGrpCrop failed with %#x\n", ret);
      return ret;
    }
  }

  for (uint32_t i = 0; i < m_enabled_chn; i++) {
    ret = CVI_VPSS_SetChnAttr(m_grpid, i, &chn_attr[i]);
    if (ret != CVI_SUCCESS) {
      syslog(LOG_ERR, "CVI_VPSS_SetChnAttr failed with %#x\n", ret);
      return ret;
    }
  }

  if (chn_crop_attr != NULL) {
    for (uint32_t i = 0; i < m_enabled_chn; i++) {
      int ret = CVI_VPSS_SetChnCrop(m_grpid, i, &chn_crop_attr[i]);
      if (ret != CVI_SUCCESS) {
        syslog(LOG_ERR, "CVI_VPSS_SetChnCrop failed with %#x\n", ret);
        return ret;
      }
    }
  }

  ret = CVI_VPSS_SendFrame(m_grpid, frame, -1);
  return ret;
}

int VpssEngine::sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                          const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, NULL, chn_attr, enable_chns);
}

int VpssEngine::sendCropGrpFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, crop_attr, NULL, chn_attr, enable_chns);
}

int VpssEngine::sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, crop_attr, chn_attr, enable_chns);
}

int VpssEngine::sendCropGrpChnFrame(const VIDEO_FRAME_INFO_S *frame,
                                    const VPSS_CROP_INFO_S *grp_crop_attr,
                                    const VPSS_CROP_INFO_S *chn_crop_attr,
                                    const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, grp_crop_attr, chn_crop_attr, chn_attr, enable_chns);
}

int VpssEngine::getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout) {
  int ret = CVI_VPSS_GetChnFrame(m_grpid, chn_idx, outframe, timeout);
  // Reset crop settings
  VPSS_CROP_INFO_S crop_attr;
  memset(&crop_attr, 0, sizeof(VPSS_CROP_INFO_S));
  CVI_VPSS_SetGrpCrop(m_grpid, &crop_attr);
  for (uint32_t i = 0; i < m_enabled_chn; i++) {
    CVI_VPSS_SetChnCrop(m_grpid, i, &crop_attr);
  }
  return ret;
}

int VpssEngine::releaseFrame(VIDEO_FRAME_INFO_S *frame, int chn_idx) {
  return CVI_VPSS_ReleaseChnFrame(m_grpid, chn_idx, frame);
}
}  // namespace cviai