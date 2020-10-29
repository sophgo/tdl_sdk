
#include "vi_vo_utils.h"
#include "core/utils/vpss_helper.h"

CVI_S32 InitVI(SAMPLE_VI_CONFIG_S *pstViConfig, CVI_U32 *devNum) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  SAMPLE_INI_CFG_S stIniCfg;
  if (!SAMPLE_COMM_VI_ParseIni(&stIniCfg)) {
    syslog(LOG_ERR | LOG_LOCAL7, "Init pasre failed.\n");
    return CVI_FAILURE;
  }
  *devNum = stIniCfg.devNum;

  DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
  PIXEL_FORMAT_E enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
  VI_VPSS_MODE_E enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;

  VI_CHN viChn = 0;
  CVI_S32 s32WorkSnsId = 0;
  PIC_SIZE_E enPicSize;
  SIZE_S stSize;

  SAMPLE_COMM_VI_GetSensorInfo(pstViConfig);
  for (; s32WorkSnsId < stIniCfg.devNum; s32WorkSnsId++) {
    pstViConfig->s32WorkingViNum = 1 + s32WorkSnsId;
    pstViConfig->as32WorkingViId[s32WorkSnsId] = s32WorkSnsId;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType =
        (s32WorkSnsId == 0) ? stIniCfg.enSnsType : stIniCfg.enSns2Type;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.MipiDev =
        (s32WorkSnsId == 0) ? stIniCfg.MipiDev : stIniCfg.Sns2MipiDev;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.s32BusId =
        (s32WorkSnsId == 0) ? stIniCfg.s32BusId : stIniCfg.s32Sns2BusId;
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[0] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[0] : stIniCfg.as16Sns2LaneId[0];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[1] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[1] : stIniCfg.as16Sns2LaneId[1];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[2] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[2] : stIniCfg.as16Sns2LaneId[2];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[3] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[3] : stIniCfg.as16Sns2LaneId[3];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as16LaneId[4] =
        (s32WorkSnsId == 0) ? stIniCfg.as16LaneId[4] : stIniCfg.as16Sns2LaneId[4];

    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[0] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[0] : stIniCfg.as8Sns2PNSwap[0];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[1] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[1] : stIniCfg.as8Sns2PNSwap[1];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[2] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[2] : stIniCfg.as8Sns2PNSwap[2];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[3] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[3] : stIniCfg.as8Sns2PNSwap[3];
    pstViConfig->astViInfo[s32WorkSnsId].stSnsInfo.as8PNSwap[4] =
        (s32WorkSnsId == 0) ? stIniCfg.as8PNSwap[4] : stIniCfg.as8Sns2PNSwap[4];

    pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.ViDev = 0;
    pstViConfig->astViInfo[s32WorkSnsId].stDevInfo.enWDRMode =
        (s32WorkSnsId == 0) ? stIniCfg.enWDRMode : stIniCfg.enSns2WDRMode;

    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0] = s32WorkSnsId;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1] = -1;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2] = -1;
    pstViConfig->astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3] = -1;

    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.ViChn = viChn;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enPixFormat = enPixFormat;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange = enDynamicRange;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat = enVideoFormat;
    pstViConfig->astViInfo[s32WorkSnsId].stChnInfo.enCompressMode = enCompressMode;
  }

  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType, &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = MMF_INIT_HELPER2(stSize.u32Width, stSize.u32Height, enPixFormat, 18, stSize.u32Width,
                            stSize.u32Height, enPixFormat, 18);
  if (s32Ret != CVI_SUCCESS) {
    printf("sys init failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_PLAT_VI_INIT(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi init failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }
  return s32Ret;
}

CVI_S32 InitVO(const CVI_U32 width, const CVI_U32 height, SAMPLE_VO_CONFIG_S *stVoConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  s32Ret = SAMPLE_COMM_VO_GetDefConfig(stVoConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_GetDefConfig failed with %#x\n", s32Ret);
    return s32Ret;
  }
  RECT_S dispRect = {0, 0, height, width};
  SIZE_S imgSize = {height, width};

  stVoConfig->VoDev = 0;
  stVoConfig->enVoIntfType = VO_INTF_MIPI;
  stVoConfig->enIntfSync = VO_OUTPUT_720x1280_60;
  stVoConfig->stDispRect = dispRect;
  stVoConfig->stImageSize = imgSize;
  stVoConfig->enPixFormat = PIXEL_FORMAT_YUV_PLANAR_420;
  stVoConfig->enVoMode = VO_MODE_1MUX;

  s32Ret = SAMPLE_COMM_VO_StartVO(stVoConfig);
  if (s32Ret != CVI_SUCCESS) {
    printf("SAMPLE_COMM_VO_StartVO failed with %#x\n", s32Ret);
  }
  CVI_VO_SetChnRotation(0, 0, ROTATION_270);
  printf("SAMPLE_COMM_VO_StartVO done\n");
  return s32Ret;
}

CVI_S32 InitVPSS(const VPSS_GRP vpssGrp, const VPSS_CHN vpssChn, const VPSS_CHN vpssChnVO,
                 const CVI_U32 grpWidth, const CVI_U32 grpHeight, const CVI_U32 voWidth,
                 const CVI_U32 voHeight, const VI_PIPE viPipe, const CVI_BOOL isVOOpened) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {0};
  VPSS_CHN_ATTR_S stVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

  abChnEnable[vpssChn] = CVI_TRUE;
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChn], voWidth, voHeight, PIXEL_FORMAT_RGB_888, true);

  if (isVOOpened) {
    abChnEnable[vpssChnVO] = CVI_TRUE;
    VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr[vpssChnVO], voWidth, voHeight,
                            PIXEL_FORMAT_YUV_PLANAR_420, true);
  }

  CVI_SYS_SetVPSSMode(VPSS_MODE_SINGLE);

  VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, grpWidth, grpHeight, PIXEL_FORMAT_YUV_PLANAR_420);

  /*start vpss*/
  s32Ret = SAMPLE_COMM_VPSS_Init(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("init vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VPSS_Start(vpssGrp, abChnEnable, &stVpssGrpAttr, stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_Bind_VPSS(viPipe, vpssChn, vpssGrp);
  if (s32Ret != CVI_SUCCESS) {
    printf("vi bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}