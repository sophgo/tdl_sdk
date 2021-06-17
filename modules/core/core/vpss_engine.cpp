#include "vpss_engine.hpp"
#include "core/utils/vpss_helper.h"
#include "cviai_log.hpp"

namespace cviai {

VpssEngine::VpssEngine() {}

VpssEngine::~VpssEngine() { stop(); }

int VpssEngine::init(VPSS_GRP grp_id, CVI_U8 device) {
  if (m_is_vpss_init) {
    LOGW("Vpss already init.\n");
    return CVI_FAILURE;
  }
  int s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_SYS_Init failed!\n");
    return s32Ret;
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
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, width, height, VI_PIXEL_FORMAT);
  VPSS_CHN_DEFAULT_HELPER(&vpss_chn_attr, width, height, PIXEL_FORMAT_RGB_888_PLANAR, true);

  /*start vpss*/
  m_grpid = -1;
  if (grp_id != (VPSS_GRP)-1) {
    if (CVI_VPSS_CreateGrp(grp_id, &vpss_grp_attr) != CVI_SUCCESS) {
      LOGE("User assign group id %u failed to create vpss instance.\n", grp_id);
      return CVI_FAILURE;
    }
    m_grpid = grp_id;
  } else {
    int id = CVI_VPSS_GetAvailableGrp();
    if (CVI_VPSS_CreateGrp(id, &vpss_grp_attr) != CVI_SUCCESS) {
      LOGE("User assign group id %u failed to create vpss instance.\n", grp_id);
      return CVI_FAILURE;
    }
    m_grpid = id;
  }
  if (m_grpid == (VPSS_GRP)-1) {
    LOGE("All vpss grp init failed!\n");
    return CVI_FAILURE;
  }
  s32Ret = CVI_VPSS_ResetGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_ResetGrp(grp:%d) failed with %#x!\n", m_grpid, s32Ret);
    return CVI_FAILURE;
  }

  for (uint32_t i = 0; i < m_enabled_chn; i++) {
    s32Ret = CVI_VPSS_SetChnAttr(m_grpid, i, &vpss_chn_attr);

    if (s32Ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
      return CVI_FAILURE;
    }

    s32Ret = CVI_VPSS_EnableChn(m_grpid, i);

    if (s32Ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
      return CVI_FAILURE;
    }
  }
  s32Ret = CVI_VPSS_StartGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return CVI_FAILURE;
  }

  m_dev = device;
  memset(&m_crop_attr_reset, 0, sizeof(VPSS_CROP_INFO_S));
  m_is_vpss_init = true;
  return CVI_SUCCESS;
}

int VpssEngine::stop() {
  if (!m_is_vpss_init) {
    LOGE("Vpss is not init yet.\n");
    return CVI_FAILURE;
  }

  for (uint32_t j = 0; j < m_enabled_chn; j++) {
    int s32Ret = CVI_VPSS_DisableChn(m_grpid, j);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_DisableChn failed with %#x!\n", s32Ret);
      return CVI_FAILURE;
    }
  }

  int s32Ret = CVI_VPSS_StopGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_StopGrp failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  s32Ret = CVI_VPSS_DestroyGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_DestroyGrp failed with %#x!\n", s32Ret);
    return CVI_FAILURE;
  }

  m_is_vpss_init = false;
  return CVI_SUCCESS;
}

VPSS_GRP VpssEngine::getGrpId() { return m_grpid; }

int VpssEngine::sendFrameBase(const VIDEO_FRAME_INFO_S *frame,
                              const VPSS_CROP_INFO_S *grp_crop_attr,
                              const VPSS_CROP_INFO_S *chn_crop_attr,
                              const VPSS_CHN_ATTR_S *chn_attr, const VPSS_SCALE_COEF_E *coeffs,
                              const uint32_t enable_chns) {
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
  VPSS_GRP_DEFAULT_HELPER2(&vpss_grp_attr, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                           frame->stVFrame.enPixelFormat, m_dev);

  if (m_enabled_chn > m_available_max_chn) {
    LOGE("Exceed max available channel %u. Current: %u.\n", m_available_max_chn, m_enabled_chn);
    return CVI_FAILURE;
  }

  int ret = CVI_VPSS_SetGrpAttr(m_grpid, &vpss_grp_attr);
  if (ret != CVI_SUCCESS) {
    LOGE("CVI_VPSS_SetGrpAttr failed with %#x\n", ret);
    return ret;
  }
  if (grp_crop_attr != NULL) {
    int ret = CVI_VPSS_SetGrpCrop(m_grpid, grp_crop_attr);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SetGrpCrop failed with %#x\n", ret);
      return ret;
    }
  } else {
    // Reset crop settings
    CVI_VPSS_SetGrpCrop(m_grpid, &m_crop_attr_reset);
  }

  for (uint32_t i = 0; i < m_enabled_chn; i++) {
    ret = CVI_VPSS_SetChnAttr(m_grpid, i, &chn_attr[i]);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_SetChnAttr failed with %#x\n", ret);
      return ret;
    }
  }

  if (chn_crop_attr != NULL) {
    for (uint32_t i = 0; i < m_enabled_chn; i++) {
      int ret = CVI_VPSS_SetChnCrop(m_grpid, i, &chn_crop_attr[i]);
      if (ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_SetChnCrop failed with %#x\n", ret);
        return ret;
      }
    }
  } else {
    for (uint32_t i = 0; i < m_enabled_chn; i++) {
      CVI_VPSS_SetChnCrop(m_grpid, i, &m_crop_attr_reset);
    }
  }

  if (coeffs != NULL) {
    for (uint32_t i = 0; i < m_enabled_chn; i++) {
      int ret = CVI_VPSS_SetChnScaleCoefLevel(m_grpid, i, coeffs[i]);
      if (ret != CVI_SUCCESS) {
        LOGE("CVI_VPSS_GetChnScaleCoefLevel failed with %#x\n", ret);
        return ret;
      }
    }
  } else {
    for (uint32_t i = 0; i < m_enabled_chn; i++) {
      // Default value
      CVI_VPSS_SetChnScaleCoefLevel(m_grpid, i, VPSS_SCALE_COEF_BICUBIC);
    }
  }

  ret = CVI_VPSS_SendFrame(m_grpid, frame, -1);
  return ret;
}

int VpssEngine::sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                          const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, NULL, chn_attr, NULL, enable_chns);
}

int VpssEngine::sendFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                          const VPSS_SCALE_COEF_E *coeffs, const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, NULL, chn_attr, coeffs, enable_chns);
}

int VpssEngine::sendCropGrpFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, crop_attr, NULL, chn_attr, NULL, enable_chns);
}

int VpssEngine::sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, crop_attr, chn_attr, NULL, enable_chns);
}

int VpssEngine::sendCropChnFrame(const VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const VPSS_SCALE_COEF_E *coeffs,
                                 const uint32_t enable_chns) {
  return sendFrameBase(frame, NULL, crop_attr, chn_attr, coeffs, enable_chns);
}

int VpssEngine::sendCropGrpChnFrame(const VIDEO_FRAME_INFO_S *frame,
                                    const VPSS_CROP_INFO_S *grp_crop_attr,
                                    const VPSS_CROP_INFO_S *chn_crop_attr,
                                    const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  return sendFrameBase(frame, grp_crop_attr, chn_crop_attr, chn_attr, NULL, enable_chns);
}

int VpssEngine::getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout) {
  int ret = CVI_VPSS_GetChnFrame(m_grpid, chn_idx, outframe, timeout);
  return ret;
}

int VpssEngine::releaseFrame(VIDEO_FRAME_INFO_S *frame, int chn_idx) {
  return CVI_VPSS_ReleaseChnFrame(m_grpid, chn_idx, frame);
}
}  // namespace cviai