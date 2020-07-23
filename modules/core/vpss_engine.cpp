#include "vpss_engine.hpp"

#include "cviruntime.h"
#include "utils/vpss_helper.h"

namespace cviai {

VpssEngine::VpssEngine() {}

VpssEngine::~VpssEngine() {}

void VpssEngine::enableLog() {
  // Tunr on Vpss Log
  LOG_LEVEL_CONF_S log_conf;
  log_conf.enModId = (MOD_ID_E)6;  // vpss
  CVI_LOG_GetLevelConf(&log_conf);
  printf("Set Vpss Log Level: %d, log will save into cvi_mmf_aisdk.log\n", log_conf.s32Level);
  log_conf.s32Level = 7;
  CVI_LOG_SetLevelConf(&log_conf);

  CVI_LOG_EnableLog2File(CVI_TRUE, (char *)"cvi_mmf_aisdk.log");
  m_enable_log = true;
}

int VpssEngine::init(bool enable_log) {
  if (m_is_vpss_init) {
    printf("Vpss already init.\n");
    return CVI_RC_FAILURE;
  }
  if (enable_log) {
    enableLog();
  }
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_CHN_ATTR_S vpss_chn_attr[VPSS_MAX_PHY_CHN_NUM];
  // Not magic number, only for init.
  uint32_t width = 100;
  uint32_t height = 100;
  PIXEL_FORMAT_E format = PIXEL_FORMAT_RGB_888_PLANAR;
  m_enabled_chn = 2;
  CVI_BOOL chn_enable[VPSS_MAX_PHY_CHN_NUM] = {0};
  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, width, height, format);
  for (uint8_t i = 0; i < m_enabled_chn; i++) {
    VPSS_CHN_DEFAULT_HELPER(&vpss_chn_attr[i], width, height, format, true);
    chn_enable[i] = CVI_TRUE;
  }

  /*start vpss*/
  m_grpid = -1;
  for (uint8_t i = 0; i < VPSS_MAX_GRP_NUM; i++) {
    int s32Ret = SAMPLE_COMM_VPSS_Init(i, chn_enable, &vpss_grp_attr, vpss_chn_attr);
    if (s32Ret == CVI_SUCCESS) {
      m_grpid = i;
      break;
    }
  }
  if (m_grpid == (CVI_U32)-1) {
    printf("All vpss grp init failed!\n");
    return CVI_RC_FAILURE;
  }

  int s32Ret = SAMPLE_COMM_VPSS_Start(m_grpid, chn_enable, &vpss_grp_attr, vpss_chn_attr);
  if (s32Ret != CVI_SUCCESS) {
    printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return CVI_RC_FAILURE;
  }
  m_is_vpss_init = true;
  return CVI_RC_SUCCESS;
}

int VpssEngine::stop() {
  if (!m_is_vpss_init) {
    printf("Vpss is not init yet.\n");
    return CVI_RC_FAILURE;
  }

  for (uint32_t j = 0; j < m_enabled_chn; j++) {
    int s32Ret = CVI_VPSS_DisableChn(m_grpid, j);
    if (s32Ret != CVI_SUCCESS) {
      printf("failed with %#x!\n", s32Ret);
      return CVI_FAILURE;
    }
  }

  int s32Ret = CVI_VPSS_StopGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", s32Ret);
    return CVI_RC_FAILURE;
  }

  s32Ret = CVI_VPSS_DestroyGrp(m_grpid);
  if (s32Ret != CVI_SUCCESS) {
    printf("failed with %#x!\n", s32Ret);
    return CVI_RC_FAILURE;
  }

  if (m_enable_log) {
    CVI_LOG_EnableLog2File(CVI_FALSE, CVI_NULL);
  }
  m_is_vpss_init = false;
  return CVI_RC_SUCCESS;
}

VPSS_GRP VpssEngine::getGrpId() { return m_grpid; }

int VpssEngine::sendFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CHN_ATTR_S *chn_attr,
                          const uint32_t enable_chns) {
  if (enable_chns > m_enabled_chn) {
    printf("Error, exceed enabled chn. Enabled: %x, required %x\n", m_enabled_chn, enable_chns);
    return CVI_RC_FAILURE;
  }
  VPSS_GRP_ATTR_S vpss_grp_attr;
  VPSS_GRP_DEFAULT_HELPER(&vpss_grp_attr, frame->stVFrame.u32Width, frame->stVFrame.u32Height,
                          frame->stVFrame.enPixelFormat);
  int ret = CVI_VPSS_SetGrpAttr(m_grpid, &vpss_grp_attr);

  for (uint32_t i = 0; i < enable_chns; i++) {
    ret = CVI_VPSS_SetChnAttr(m_grpid, i, &chn_attr[i]);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VPSS_SetChnAttr failed with %#x\n", ret);
      return CVI_RC_FAILURE;
    }
  }

  CVI_VPSS_SendFrame(m_grpid, frame, -1);
  return CVI_RC_SUCCESS;
}

int VpssEngine::sendCropGrpFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  int ret = CVI_VPSS_SetGrpCrop(m_grpid, crop_attr);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SetGrpCrop failed with %#x\n", ret);
    return CVI_RC_FAILURE;
  }
  return sendFrame(frame, chn_attr, enable_chns);
}

int VpssEngine::sendCropChnFrame(VIDEO_FRAME_INFO_S *frame, const VPSS_CROP_INFO_S *crop_attr,
                                 const VPSS_CHN_ATTR_S *chn_attr, const uint32_t enable_chns) {
  if (enable_chns > m_enabled_chn) {
    printf("Error, exceed enabled chn. Enabled: %x, required %x\n", m_enabled_chn, enable_chns);
    return CVI_RC_FAILURE;
  }

  for (uint32_t i = 0; i < enable_chns; i++) {
    int ret = CVI_VPSS_SetChnCrop(m_grpid, i, &crop_attr[i]);
    if (ret != CVI_SUCCESS) {
      printf("CVI_VPSS_SetChnCrop failed with %#x\n", ret);
      return CVI_RC_FAILURE;
    }
  }
  return sendFrame(frame, chn_attr, enable_chns);
}

int VpssEngine::getFrame(VIDEO_FRAME_INFO_S *outframe, int chn_idx, uint32_t timeout) {
  int ret = CVI_VPSS_GetChnFrame(m_grpid, chn_idx, outframe, timeout);
  if (ret != CVI_SUCCESS) {
    printf("CVI_VPSS_GetChnFrame failed with %#x\n", ret);
    return CVI_RC_FAILURE;
  }
  return CVI_RC_SUCCESS;
}
}  // namespace cviai