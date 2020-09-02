#pragma once
#include <cvi_buffer.h>
#include <cvi_comm_vb.h>
#include <cvi_comm_vpss.h>

#include <cvi_math.h>
#include <cvi_sys.h>
#include <cvi_vb.h>
#include <cvi_vpss.h>

#include <math.h>
#include <string.h>

#define VIP_WIDTH_ALIGN 32
#define SCALAR_4096_ALIGN_BUG 0x1000

// Legacy support
static inline int __attribute__((always_inline))
MMF_INIT_HELPER2(uint32_t enSrcWidth, uint32_t enSrcHeight, PIXEL_FORMAT_E enSrcFormat,
                 const uint32_t inBlkCount, uint32_t enDstWidth, uint32_t enDstHeight,
                 PIXEL_FORMAT_E enDstFormat, const uint32_t outBlkCount) {
  COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
  // Init SYS and Common VB,
  // Running w/ Vi don't need to do it again. Running Vpss along need init below
  // FIXME: Can only be init once in one pipeline
  VB_CONFIG_S stVbConf;
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = 2;
  CVI_U32 u32BlkSize;
  u32BlkSize = COMMON_GetPicBufferSize(enSrcWidth, enSrcHeight, enSrcFormat, DATA_BITWIDTH_8,
                                       enCompressMode, DEFAULT_ALIGN);
  stVbConf.astCommPool[0].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[0].u32BlkCnt = inBlkCount;
  u32BlkSize = COMMON_GetPicBufferSize(enDstWidth, enDstHeight, enDstFormat, DATA_BITWIDTH_8,
                                       enCompressMode, DEFAULT_ALIGN);
  stVbConf.astCommPool[1].u32BlkSize = u32BlkSize;
  stVbConf.astCommPool[1].u32BlkCnt = outBlkCount;

  CVI_S32 s32Ret = CVI_FAILURE;

  CVI_SYS_Exit();
  CVI_VB_Exit();

  s32Ret = CVI_VB_SetConfig(&stVbConf);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VB_SetConf failed!\n");
    return s32Ret;
  }
  s32Ret = CVI_VB_Init();
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VB_Init failed!\n");
    return s32Ret;
  }
  s32Ret = CVI_SYS_Init();
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_SYS_Init failed!\n");
    CVI_VB_Exit();
    return s32Ret;
  }

  return s32Ret;
}

static inline int __attribute__((always_inline))
MMF_INIT_HELPER(uint32_t enSrcWidth, uint32_t enSrcHeight, PIXEL_FORMAT_E enSrcFormat,
                uint32_t enDstWidth, uint32_t enDstHeight, PIXEL_FORMAT_E enDstFormat) {
  return MMF_INIT_HELPER2(enSrcWidth, enSrcHeight, enSrcFormat, 12, enDstWidth, enDstHeight,
                          enDstFormat, 12);
}

inline void __attribute__((always_inline))
VPSS_GRP_DEFAULT_HELPER(VPSS_GRP_ATTR_S *pstVpssGrpAttr, CVI_U32 srcWidth, CVI_U32 srcHeight,
                        PIXEL_FORMAT_E enSrcFormat) {
  memset(pstVpssGrpAttr, 0, sizeof(VPSS_GRP_ATTR_S));
  pstVpssGrpAttr->stFrameRate.s32SrcFrameRate = -1;
  pstVpssGrpAttr->stFrameRate.s32DstFrameRate = -1;
  pstVpssGrpAttr->enPixelFormat = enSrcFormat;
  pstVpssGrpAttr->u32MaxW = srcWidth;
  pstVpssGrpAttr->u32MaxH = srcHeight;
  pstVpssGrpAttr->u8VpssDev = 0;
}

inline void __attribute__((always_inline))
VPSS_CHN_DEFAULT_HELPER(VPSS_CHN_ATTR_S *pastVpssChnAttr, CVI_U32 dstWidth, CVI_U32 dstHeight,
                        PIXEL_FORMAT_E enDstFormat, CVI_BOOL keepAspectRatio) {
  pastVpssChnAttr->u32Width = dstWidth;
  pastVpssChnAttr->u32Height = dstHeight;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = enDstFormat;

  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;
  if (keepAspectRatio) {
    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_AUTO;
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  } else {
    pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_NONE;
  }
  pastVpssChnAttr->stNormalize.bEnable = CVI_FALSE;
  pastVpssChnAttr->stNormalize.factor[0] = 0;
  pastVpssChnAttr->stNormalize.factor[1] = 0;
  pastVpssChnAttr->stNormalize.factor[2] = 0;
  pastVpssChnAttr->stNormalize.mean[0] = 0;
  pastVpssChnAttr->stNormalize.mean[1] = 0;
  pastVpssChnAttr->stNormalize.mean[2] = 0;
  pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
}

inline void __attribute__((always_inline))
VPSS_CHN_RATIO_MANUAL_HELPER(VPSS_CHN_ATTR_S *pastVpssChnAttr, CVI_U32 dstWidth, CVI_U32 dstHeight,
                             PIXEL_FORMAT_E enDstFormat, CVI_U32 ratioX, CVI_U32 ratioY,
                             CVI_U32 ratioWidth, CVI_U32 ratioHeight) {
  pastVpssChnAttr->u32Width = dstWidth;
  pastVpssChnAttr->u32Height = dstHeight;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = enDstFormat;

  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;
  pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
  pastVpssChnAttr->stAspectRatio.stVideoRect.s32Y = ratioX;
  pastVpssChnAttr->stAspectRatio.stVideoRect.s32X = ratioY;
  pastVpssChnAttr->stAspectRatio.stVideoRect.u32Width = ratioWidth;
  pastVpssChnAttr->stAspectRatio.stVideoRect.u32Height = ratioHeight;
  pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  pastVpssChnAttr->stNormalize.bEnable = CVI_FALSE;
  pastVpssChnAttr->stNormalize.factor[0] = 0;
  pastVpssChnAttr->stNormalize.factor[1] = 0;
  pastVpssChnAttr->stNormalize.factor[2] = 0;
  pastVpssChnAttr->stNormalize.mean[0] = 0;
  pastVpssChnAttr->stNormalize.mean[1] = 0;
  pastVpssChnAttr->stNormalize.mean[2] = 0;
  pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
}

inline void __attribute__((always_inline))
VPSS_CHN_SQ_HELPER(VPSS_CHN_ATTR_S *pastVpssChnAttr, const CVI_U32 dstWidth,
                   const CVI_U32 dstHeight, const PIXEL_FORMAT_E enDstFormat,
                   const CVI_FLOAT *factor, const CVI_FLOAT *mean, const bool padReverse) {
  pastVpssChnAttr->u32Width = dstWidth;
  pastVpssChnAttr->u32Height = dstHeight;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = enDstFormat;
  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;
  pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_AUTO;
  pastVpssChnAttr->stAspectRatio.bEnableBgColor = CVI_TRUE;
  if (padReverse) {
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(
        (int)(mean[0] / factor[0]), (int)(mean[1] / factor[1]), (int)(mean[2] / factor[2]));
  } else {
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  }
  pastVpssChnAttr->stNormalize.bEnable = CVI_TRUE;
  for (uint32_t i = 0; i < 3; i++) {
    pastVpssChnAttr->stNormalize.factor[i] = factor[i];
  }
  for (uint32_t i = 0; i < 3; i++) {
    pastVpssChnAttr->stNormalize.mean[i] = mean[i];
  }
  pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
}

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#define min(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

inline void __attribute__((always_inline))
VPSS_CHN_SQ_RB_HELPER(VPSS_CHN_ATTR_S *pastVpssChnAttr, const CVI_U32 srcWidth,
                      const CVI_U32 srcHeight, const CVI_U32 dstWidth, const CVI_U32 dstHeight,
                      const PIXEL_FORMAT_E enDstFormat, const CVI_FLOAT *factor,
                      const CVI_FLOAT *mean, const bool padReverse, const bool smallRatioMajor) {
  float ratio_w = (float)dstWidth / srcWidth;
  float ratio_h = (float)dstHeight / srcHeight;
  float ratio = smallRatioMajor ? min(ratio_w, ratio_h) : max(ratio_w, ratio_h);

  pastVpssChnAttr->u32Width = dstWidth;
  pastVpssChnAttr->u32Height = dstHeight;
  pastVpssChnAttr->enVideoFormat = VIDEO_FORMAT_LINEAR;
  pastVpssChnAttr->enPixelFormat = enDstFormat;
  pastVpssChnAttr->stFrameRate.s32SrcFrameRate = -1;
  pastVpssChnAttr->stFrameRate.s32DstFrameRate = -1;
  pastVpssChnAttr->u32Depth = 1;
  pastVpssChnAttr->bMirror = CVI_FALSE;
  pastVpssChnAttr->bFlip = CVI_FALSE;
  pastVpssChnAttr->stAspectRatio.enMode = ASPECT_RATIO_MANUAL;
  pastVpssChnAttr->stAspectRatio.stVideoRect.s32X = 0;
  pastVpssChnAttr->stAspectRatio.stVideoRect.s32Y = 0;
  pastVpssChnAttr->stAspectRatio.stVideoRect.u32Width = (srcWidth * ratio) + 0.5;
  pastVpssChnAttr->stAspectRatio.stVideoRect.u32Height = (srcHeight * ratio) + 0.5;
  pastVpssChnAttr->stAspectRatio.bEnableBgColor = CVI_TRUE;
  if (padReverse) {
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(
        (int)(mean[0] / factor[0]), (int)(mean[1] / factor[1]), (int)(mean[2] / factor[2]));
  } else {
    pastVpssChnAttr->stAspectRatio.u32BgColor = RGB_8BIT(0, 0, 0);
  }
  pastVpssChnAttr->stNormalize.bEnable = CVI_TRUE;
  for (uint32_t i = 0; i < 3; i++) {
    pastVpssChnAttr->stNormalize.factor[i] = factor[i];
  }
  for (uint32_t i = 0; i < 3; i++) {
    pastVpssChnAttr->stNormalize.mean[i] = mean[i];
  }
  pastVpssChnAttr->stNormalize.rounding = VPSS_ROUNDING_TO_EVEN;
}

#undef max
#undef min

inline int __attribute__((always_inline))
VPSS_INIT_HELPER(CVI_U32 VpssGrpId, uint32_t enSrcWidth, uint32_t enSrcHeight, uint32_t enSrcStride,
                 PIXEL_FORMAT_E enSrcFormat, uint32_t enDstWidth, uint32_t enDstHeight,
                 PIXEL_FORMAT_E enDstFormat, VPSS_MODE_E mode, bool keepAspectRatio,
                 bool enableLog) {
  printf("VPSS init with src (%u, %u) dst (%u, %u).\n", enSrcWidth, enSrcHeight, enDstWidth,
         enDstHeight);
  CVI_S32 s32Ret = CVI_FAILURE;

  // Tunr on Vpss Log
  if (enableLog) {
    LOG_LEVEL_CONF_S log_conf;
    log_conf.enModId = (MOD_ID_E)6;  // vpss
    CVI_LOG_GetLevelConf(&log_conf);
    printf("Set Vpss Log Level: %d, log will save into cvi_mmf.log\n", log_conf.s32Level);
    log_conf.s32Level = 7;
    CVI_LOG_SetLevelConf(&log_conf);

    log_conf.enModId = (MOD_ID_E)14;  // VI
    CVI_LOG_GetLevelConf(&log_conf);
    printf("Set VI Log Level: %d, log will save into cvi_mmf.log\n", log_conf.s32Level);
    log_conf.s32Level = 7;
    CVI_LOG_SetLevelConf(&log_conf);
    CVI_LOG_EnableLog2File(CVI_TRUE, (char *)"cvi_mmf.log");
  }

  CVI_SYS_SetVPSSMode(mode);
  VPSS_GRP_ATTR_S stVpssGrpAttr;
  VPSS_CHN_ATTR_S stVpssChnAttr;

  VPSS_GRP_DEFAULT_HELPER(&stVpssGrpAttr, enSrcWidth, enSrcHeight, enSrcFormat);
  VPSS_CHN_DEFAULT_HELPER(&stVpssChnAttr, enDstWidth, enDstHeight, enDstFormat, keepAspectRatio);

  /*start vpss*/
  s32Ret = CVI_VPSS_CreateGrp(VpssGrpId, &stVpssGrpAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_CreateGrp(grp:%d) failed with %#x!\n", VpssGrpId, s32Ret);
    return s32Ret;
  }
  s32Ret = CVI_VPSS_ResetGrp(VpssGrpId);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_ResetGrp(grp:%d) failed with %#x!\n", VpssGrpId, s32Ret);
    return s32Ret;
  }
  s32Ret = CVI_VPSS_SetChnAttr(VpssGrpId, VPSS_CHN0, &stVpssChnAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_SetChnAttr failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = CVI_VPSS_EnableChn(VpssGrpId, VPSS_CHN0);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_EnableChn failed with %#x\n", s32Ret);
    return s32Ret;
  }
  s32Ret = CVI_VPSS_StartGrp(VpssGrpId);
  if (s32Ret != CVI_SUCCESS) {
    printf("CVI_VPSS_StartGrp failed with %#x\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}

inline int __attribute__((always_inline))
CREATE_VBFRAME_HELPER(VB_BLK *blk, VIDEO_FRAME_INFO_S *vbFrame, CVI_U32 srcWidth, CVI_U32 srcHeight,
                      PIXEL_FORMAT_E pixelFormat) {
  // Create Src Video Frame
  VIDEO_FRAME_S *vFrame = &vbFrame->stVFrame;
  vFrame->enCompressMode = COMPRESS_MODE_NONE;
  vFrame->enPixelFormat = pixelFormat;
  vFrame->enVideoFormat = VIDEO_FORMAT_LINEAR;
  vFrame->enColorGamut = COLOR_GAMUT_BT709;
  vFrame->u32TimeRef = 0;
  vFrame->u64PTS = 0;
  vFrame->enDynamicRange = DYNAMIC_RANGE_SDR8;

  vFrame->u32Width = srcWidth;
  vFrame->u32Height = srcHeight;
  switch (vFrame->enPixelFormat) {
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_BGR_888: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width * 3, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      // Don't need to align cause only 1 chn.
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
    } break;
    case PIXEL_FORMAT_RGB_888_PLANAR: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[1] = vFrame->u32Stride[0];
      vFrame->u32Stride[2] = vFrame->u32Stride[0];
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = vFrame->u32Length[0];
      vFrame->u32Length[2] = vFrame->u32Length[0];
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_422: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width >> 1, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[2] = ALIGN(vFrame->u32Width >> 1, VIP_WIDTH_ALIGN);
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * vFrame->u32Height, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = ALIGN(vFrame->u32Stride[1] * vFrame->u32Height, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] = ALIGN(vFrame->u32Stride[2] * vFrame->u32Height, SCALAR_4096_ALIGN_BUG);
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_420: {
      uint32_t newHeight = ALIGN(vFrame->u32Height, 2);
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[1] = ALIGN(vFrame->u32Width >> 1, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[2] = ALIGN(vFrame->u32Width >> 1, VIP_WIDTH_ALIGN);
      vFrame->u32Length[0] = ALIGN(vFrame->u32Stride[0] * newHeight, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[1] = ALIGN(vFrame->u32Stride[1] * newHeight / 2, SCALAR_4096_ALIGN_BUG);
      vFrame->u32Length[2] = ALIGN(vFrame->u32Stride[2] * newHeight / 2, SCALAR_4096_ALIGN_BUG);
    } break;
    case PIXEL_FORMAT_YUV_400: {
      vFrame->u32Stride[0] = ALIGN(vFrame->u32Width, VIP_WIDTH_ALIGN);
      vFrame->u32Stride[1] = 0;
      vFrame->u32Stride[2] = 0;
      vFrame->u32Length[0] = vFrame->u32Stride[0] * vFrame->u32Height;
      vFrame->u32Length[1] = 0;
      vFrame->u32Length[2] = 0;
    } break;
    default:
      printf("Currently unsupported format %u\n", vFrame->enPixelFormat);
      return CVI_FAILURE;
      break;
  }

  CVI_U32 u32MapSize = vFrame->u32Length[0] + vFrame->u32Length[1] + vFrame->u32Length[2];
  *blk = CVI_VB_GetBlock(VB_INVALID_POOLID, u32MapSize);
  if (*blk == VB_INVALID_HANDLE) {
    printf("Can't acquire vb block Size: %d\n", u32MapSize);
    return CVI_FAILURE;
  }
  vbFrame->u32PoolId = CVI_VB_Handle2PoolId(*blk);
  vFrame->u64PhyAddr[0] = CVI_VB_Handle2PhysAddr(*blk);
  vFrame->u64PhyAddr[1] = vFrame->u64PhyAddr[0] + vFrame->u32Length[0];
  vFrame->u64PhyAddr[2] = vFrame->u64PhyAddr[1] + vFrame->u32Length[1];

  vFrame->pu8VirAddr[0] = (uint8_t *)CVI_SYS_Mmap(vFrame->u64PhyAddr[0], u32MapSize);
  vFrame->pu8VirAddr[1] = vFrame->pu8VirAddr[0] + vFrame->u32Length[0];
  vFrame->pu8VirAddr[2] = vFrame->pu8VirAddr[1] + vFrame->u32Length[1];

  return CVI_SUCCESS;
}
