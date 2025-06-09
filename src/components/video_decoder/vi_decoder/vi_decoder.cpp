#include "vi_decoder/vi_decoder.hpp"
#include "image/base_image.hpp"
#include "memory/cvi_memory_pool.hpp"
#include "sample_comm.h"
#include "utils/tdl_log.hpp"

static SAMPLE_VI_CONFIG_S g_stViConfig;
static VIDEO_FRAME_INFO_S frame_info[VI_MAX_CHN_NUM];
static VI_CHN_ATTR_S CHN_ATTR_420_SDR8 = {
    {1920, 1080},       PIXEL_FORMAT_YUV_PLANAR_420,
    DYNAMIC_RANGE_SDR8, VIDEO_FORMAT_LINEAR,
    COMPRESS_MODE_NONE, CVI_FALSE,
    CVI_FALSE,          0,
    {-1, -1},           0xFFFFFFFF,
#ifdef __CV184X__
    CVI_FALSE,
#endif
};

static CVI_S32 TDL_VI_StartViChn(SAMPLE_VI_CONFIG_S *pstViConfig,
                                 CVI_S32 *pool_id) {
  CVI_S32 i;
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 ViNum = 0;
  VI_PIPE ViPipe = 0;
  VI_CHN ViChn = 0;
  VI_DEV ViDev = 0;
  CVI_U32 u32SnsId = 0;
  VI_DEV_ATTR_S stViDevAttr;
  VI_CHN_ATTR_S stChnAttr;

#ifdef __CV184X__
  ViNum = pstViConfig->s32ViNum;
#else
  ViNum = pstViConfig->s32WorkingViNum;
#endif

  for (i = 0; i < ViNum; i++) {
    if (i < VI_MAX_DEV_NUM) {
      ViPipe = pstViConfig->astViInfo[i].stPipeInfo.aPipe[0];
      ViChn = pstViConfig->astViInfo[i].stChnInfo.ViChn;
      ViDev = pstViConfig->astViInfo[i].stDevInfo.ViDev;
      u32SnsId = pstViConfig->astViInfo[i].stSnsInfo.s32SnsId;

#ifndef __CV184X__
      ISP_SNS_OBJ_S *pstSnsObj;
      pstSnsObj = (ISP_SNS_OBJ_S *)SAMPLE_COMM_ISP_GetSnsObj(u32SnsId);
#endif

      memcpy(&stChnAttr, &CHN_ATTR_420_SDR8, sizeof(VI_CHN_ATTR_S));
      SAMPLE_COMM_VI_GetDevAttrBySns(
          pstViConfig->astViInfo[i].stSnsInfo.enSnsType, &stViDevAttr);

      if (stViDevAttr.enInputDataType == VI_DATA_TYPE_YUV) {
        stChnAttr.enPixelFormat = PIXEL_FORMAT_YUV_PLANAR_422;
      }

      stChnAttr.stSize.u32Width = stViDevAttr.stSize.u32Width;
      stChnAttr.stSize.u32Height = stViDevAttr.stSize.u32Height;
      stChnAttr.enDynamicRange =
          pstViConfig->astViInfo[i].stChnInfo.enDynamicRange;
      stChnAttr.enVideoFormat =
          pstViConfig->astViInfo[i].stChnInfo.enVideoFormat;
      stChnAttr.enCompressMode =
          pstViConfig->astViInfo[i].stChnInfo.enCompressMode;
      stChnAttr.enPixelFormat = pstViConfig->astViInfo[i].stChnInfo.enPixFormat;
      /* fill the sensor orientation */
      if (pstViConfig->astViInfo[i].stSnsInfo.u8Orien <= 3) {
        stChnAttr.bMirror = pstViConfig->astViInfo[i].stSnsInfo.u8Orien & 0x1;
        stChnAttr.bFlip = pstViConfig->astViInfo[i].stSnsInfo.u8Orien & 0x2;
      }

      s32Ret = CVI_VI_SetChnAttr(ViPipe, ViChn, &stChnAttr);
      if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VI_SetChnAttr failed with %#x!\n", s32Ret);
        return CVI_FAILURE;
      }

#ifndef __CV184X__
      if (pstSnsObj && pstSnsObj->pfnMirrorFlip) {
        CVI_VI_RegChnFlipMirrorCallBack(ViPipe, ViDev,
                                        (void *)pstSnsObj->pfnMirrorFlip);
      }
#endif

      s32Ret = CVI_VI_AttachVbPool(ViPipe, ViChn, pool_id[i]);
      if (s32Ret != 0) {
        printf("CVI_VI_AttachVbPool(%d) fail with %d", i, s32Ret);
        return CVI_FAILURE;
      }

      s32Ret = CVI_VI_EnableChn(ViPipe, ViChn);
      if (s32Ret != CVI_SUCCESS) {
        LOGE("CVI_VI_EnableChn failed with %#x!\n", s32Ret);
        return CVI_FAILURE;
      }
    }
  }

  return s32Ret;
}

static CVI_S32 TDL_VI_INIT(SAMPLE_VI_CONFIG_S *pstViConfig, CVI_S32 *pool_id) {
  PIC_SIZE_E enPicSize;
  SIZE_S stSize;

  VI_DEV ViDev = 0;
  VI_PIPE ViPipe = 0;
  VI_PIPE_ATTR_S stPipeAttr;

  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i = 0, j = 0, ViNum = 0;
  CVI_S32 s32DevNum;

  /************************************************
   * step1:  Get input size
   ************************************************/
  s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(
      pstViConfig->astViInfo[ViDev].stSnsInfo.enSnsType, &enPicSize);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", s32Ret);
    return s32Ret;
  }

  /************************************************
   * step2:  Init VI ISP
   ************************************************/
#ifndef __CV184X__
  s32Ret = SAMPLE_COMM_VI_StartSensor(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("system start sensor failed with %#x\n", s32Ret);
    return s32Ret;
  }
#endif

#ifdef __CV184X__
  ViNum = pstViConfig->s32ViNum;
#else
  ViNum = pstViConfig->s32WorkingViNum;
#endif

  s32Ret = SAMPLE_COMM_VI_StartMIPI(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("system start MIPI failed with %#x\n", s32Ret);
    return s32Ret;
  }

  s32Ret = SAMPLE_COMM_VI_SensorProbe(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("system sensor probe failed with %#x\n", s32Ret);
    return s32Ret;
  }

  for (i = 0; i < ViNum; i++) {
    ViDev = i;
    s32Ret = SAMPLE_COMM_VI_StartDev(&pstViConfig->astViInfo[ViDev]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("VI_StartDev failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }

  stPipeAttr.bYuvSkip = CVI_FALSE;
  stPipeAttr.u32MaxW = stSize.u32Width;
  stPipeAttr.u32MaxH = stSize.u32Height;
  stPipeAttr.enPixFmt = PIXEL_FORMAT_RGB_BAYER_12BPP;
  stPipeAttr.enBitWidth = DATA_BITWIDTH_12;
  stPipeAttr.stFrameRate.s32SrcFrameRate = -1;
  stPipeAttr.stFrameRate.s32DstFrameRate = -1;
  stPipeAttr.bNrEn = CVI_TRUE;
  stPipeAttr.bYuvBypassPath = CVI_FALSE;
  stPipeAttr.enCompressMode =
      pstViConfig->astViInfo[0].stChnInfo.enCompressMode;

  for (i = 0; i < ViNum; i++) {
    SAMPLE_VI_INFO_S *pstViInfo = NULL;

    s32DevNum = pstViConfig->as32WorkingViId[i];
    pstViInfo = &pstViConfig->astViInfo[s32DevNum];
    stPipeAttr.bYuvBypassPath =
        SAMPLE_COMM_VI_GetYuvBypassSts(pstViInfo->stSnsInfo.enSnsType);

    for (j = 0; j < WDR_MAX_PIPE_NUM; j++) {
      if (pstViInfo->stPipeInfo.aPipe[j] >= 0 &&
          pstViInfo->stPipeInfo.aPipe[j] < VI_MAX_PIPE_NUM) {
        ViPipe = pstViInfo->stPipeInfo.aPipe[j];
        s32Ret = CVI_VI_CreatePipe(ViPipe, &stPipeAttr);
        if (s32Ret != CVI_SUCCESS) {
          LOGE("CVI_VI_CreatePipe failed with %#x!\n", s32Ret);
          return s32Ret;
        }

        s32Ret = CVI_VI_StartPipe(ViPipe);
        if (s32Ret != CVI_SUCCESS) {
          LOGE("CVI_VI_StartPipe failed with %#x!\n", s32Ret);
          return s32Ret;
        }
      }
    }
  }

#ifndef __CV184X__
  s32Ret = SAMPLE_COMM_VI_CreateIsp(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("VI_CreateIsp failed with %#x!\n", s32Ret);
    return s32Ret;
  }
#endif

#ifdef __CV184X__
  for (i = 0; i < ViNum; i++) {
    if (CVI_SNS_SetSnsInit(i) != CVI_SUCCESS) {
      printf("[ERROR] sensor_%d init failed!\n", i);
      return CVI_FAILURE;
    }
  }
#endif

  s32Ret = TDL_VI_StartViChn(pstViConfig, pool_id);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("VI_StartViChn failed with %#x!\n", s32Ret);
    return s32Ret;
  }

  return s32Ret;
}

int ViDecoder::initialize() {
  int ret = 0;

#ifdef __CV184X__
  SNS_INI_CFG_S stIniCfg;
#else
  SAMPLE_INI_CFG_S stIniCfg;
#endif

  CVI_S32 ViNum = 0;
  SAMPLE_VI_CONFIG_S stViConfig;
  VB_CONFIG_S stVbConf;
  PIC_SIZE_E enPicSize;
  SIZE_S stSize;
  CVI_U32 u32BlkSize, u32BlkRotSize;
  CVI_U32 Vb_cnt;
  CVI_S32 pool_id[VI_MAX_PIPE_NUM] = {};

/************************************************
 * step1:  Config VI
 ************************************************/
#ifdef __CV184X__
  ret = CVI_SYS_Init();
  if (ret != 0) {
    LOGE("CVI_SYS_Init failed!\n");
    return ret;
  }

  ret = SAMPLE_COMM_VI_INI_INIT(&stViConfig, &stIniCfg, &stVbConf);
  if (ret != 0) {
    LOGE("SAMPLE_COMM_VI_INI_INIT fail\n");
    return ret;
  }
#else
  ret = SAMPLE_COMM_VI_ParseIni(&stIniCfg);
  if (ret != 0) {
    LOGE("Parse sensor_cfg.ini fail\n");
    return ret;
  }

  ret = SAMPLE_COMM_VI_IniToViCfg(&stIniCfg, &stViConfig);
  if (ret != 0) {
    LOGE("SAMPLE_COMM_VI_IniToViCfg fail\n");
    return ret;
  }
#endif

  // Set sensor number
  CVI_VI_SetDevNum(stIniCfg.devNum);

  memcpy(&g_stViConfig, &stViConfig, sizeof(SAMPLE_VI_CONFIG_S));

  /************************************************
   * step2:  Set memory_pool and memory_blocks
   ************************************************/
  memset(&stVbConf, 0, sizeof(VB_CONFIG_S));
  stVbConf.u32MaxPoolCnt = 0;

  ret = SAMPLE_COMM_SYS_Init(&stVbConf);
  if (ret != 0) {
    LOGE("SAMPLE_COMM_SYS_Init failed. ret: 0x%x !\n", ret);
    return ret;
  }

  memory_pool_ = BaseMemoryPoolFactory::createMemoryPool();
  memory_blocks_.clear();
  auto pool = std::dynamic_pointer_cast<CviMemoryPool>(memory_pool_);
  if (!pool) {
    printf("memory_pool is nullptr");
    return -1;
  }

#ifdef __CV184X__
  ViNum = stViConfig.s32ViNum;
#else
  ViNum = stViConfig.s32WorkingViNum;
#endif

  for (CVI_S32 i = 0; i < ViNum; i++) {
    ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType[i], &enPicSize);
    if (ret != CVI_SUCCESS) {
      printf("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", ret);
      return ret;
    }

    ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
    if (ret != CVI_SUCCESS) {
      printf("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", ret);
      return ret;
    }

    u32BlkSize = COMMON_GetPicBufferSize(
        stSize.u32Width, stSize.u32Height,
        stViConfig.astViInfo[i].stChnInfo.enPixFormat, DATA_BITWIDTH_8,
        COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    u32BlkRotSize = COMMON_GetPicBufferSize(
        stSize.u32Height, stSize.u32Width,
        stViConfig.astViInfo[i].stChnInfo.enPixFormat, DATA_BITWIDTH_8,
        COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    u32BlkSize = u32BlkSize > u32BlkRotSize ? u32BlkSize : u32BlkRotSize;

    memory_blocks_.push_back(
        pool->CreateExVb(u32BlkSize, 3, stSize.u32Width, stSize.u32Height));
    pool_id[i] = memory_blocks_.back()->id;
  }

  /************************************************
   * step3:  Init modules
   ************************************************/
  ret = TDL_VI_INIT(&stViConfig, pool_id);
  if (ret != 0) {
    LOGE("TDL_VI_INIT failed. ret: 0x%x !\n", ret);
    return ret;
  }

  isInitialized = true;

  return ret;
}

int ViDecoder::deinitialize() {
  int ret = 0;

  auto pool = std::dynamic_pointer_cast<CviMemoryPool>(memory_pool_);

  SAMPLE_COMM_VI_DestroyIsp(&g_stViConfig);
  SAMPLE_COMM_VI_DestroyVi(&g_stViConfig);

  if (pool) {
    for (auto &block : memory_blocks_) {
      pool->DestroyExVb(block);
    }
  }
  memory_blocks_.clear();
  memory_pool_.reset();

  SAMPLE_COMM_SYS_Exit();

  isInitialized = false;
  return ret;
}

ViDecoder::ViDecoder() {
  type_ = VideoDecoderType::VI;
  if (!isInitialized) {
    initialize();
  }
}

ViDecoder::~ViDecoder() {
  if (isInitialized) {
    deinitialize();
  }
  if (isMapped_) {
    CVI_SYS_Munmap((void *)addr_, image_size_);
  }
}

int32_t ViDecoder::init(const std::string &path,
                        const std::map<std::string, int> &config) {
  path_ = path;

  if (!isInitialized) {
    return initialize();
  }

  return 0;
}

int32_t ViDecoder::read(std::shared_ptr<BaseImage> &image, int vi_chn) {
  if (isMapped_) {
    CVI_SYS_Munmap((void *)addr_, image_size_);
    isMapped_ = false;
  }
  int ret = 0;

  while (true) {
    ret = CVI_VI_GetChnFrame(vi_chn, vi_chn, &frame_info[vi_chn], 3000);
    VIDEO_FRAME_INFO_S *vpss_frame_info = &frame_info[vi_chn];
    if (ret != 0 ||
        vpss_frame_info->stVFrame.u32Width == 0) {  // CVI_VI_GetChnFrame bug
      printf("CVI_VI_GetChnFrame(%d) failed, ret(%d) width %d\n", vi_chn, ret,
             vpss_frame_info->stVFrame.u32Width);
    } else {
      break;
    }
  }

  // 计算总的图像大小
  size_t image_size = frame_info[vi_chn].stVFrame.u32Length[0] +
                      frame_info[vi_chn].stVFrame.u32Length[1] +
                      frame_info[vi_chn].stVFrame.u32Length[2];

  // 如果虚拟地址为空，进行内存映射
  if (frame_info[vi_chn].stVFrame.pu8VirAddr[0] == NULL) {
    frame_info[vi_chn].stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_Mmap(
        frame_info[vi_chn].stVFrame.u64PhyAddr[0], image_size);
    isMapped_ = true;
    addr_ = frame_info[vi_chn].stVFrame.pu8VirAddr[0];
    image_size_ = image_size;
  }

  image = ImageFactory::wrapVPSSFrame(&frame_info[vi_chn], false);

  return image ? 0 : -1;
}

int32_t ViDecoder::read(VIDEO_FRAME_INFO_S *frame, int vi_chn) {
  int ret = 0;
  ret = CVI_VI_GetChnFrame(vi_chn, vi_chn, frame, 3000);
  if (ret != 0) {
    LOGE("CVI_VI_GetChnFrame(%d) failed with %d\n", vi_chn, ret);
    return ret;
  }

  return frame ? 0 : -1;
}

int32_t ViDecoder::release(int vi_chn) {
  int ret = 0;
  ret = CVI_VI_ReleaseChnFrame(vi_chn, vi_chn, &frame_info[vi_chn]);
  if (ret != 0) {
    LOGE("CVI_VI_ReleaseChnFrame(%d) failed with %d\n", vi_chn, ret);
  }
  return ret;
}

int32_t ViDecoder::release(int vi_chn, VIDEO_FRAME_INFO_S *frame) {
  int ret = 0;
  ret = CVI_VI_ReleaseChnFrame(vi_chn, vi_chn, frame);
  if (ret != 0) {
    LOGE("CVI_VI_ReleaseChnFrame(%d) failed with %d\n", vi_chn, ret);
  }
  return ret;
}