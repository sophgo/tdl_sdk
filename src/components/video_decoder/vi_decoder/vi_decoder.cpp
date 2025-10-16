#include "vi_decoder/vi_decoder.hpp"
#include <mutex>
#include <queue>
#include <vector>
#include "image/base_image.hpp"
#include "memory/cvi_memory_pool.hpp"
#include "sample_comm.h"
#include "utils/tdl_log.hpp"

static SAMPLE_VI_CONFIG_S g_stViConfig = {};
std::vector<std::queue<std::shared_ptr<VIDEO_FRAME_INFO_S>>> frameQueues(
    VI_MAX_PIPE_NUM);
std::vector<std::mutex> queueMutexes(VI_MAX_PIPE_NUM);

static PIXEL_FORMAT_E convertPixelFormat(ImageFormat img_format) {
  PIXEL_FORMAT_E pixel_format = PIXEL_FORMAT_MAX;

  if (img_format == ImageFormat::GRAY) {
    pixel_format = PIXEL_FORMAT_YUV_400;
  } else if (img_format == ImageFormat::YUV420SP_UV) {
    pixel_format = PIXEL_FORMAT_NV12;
  } else if (img_format == ImageFormat::YUV420SP_VU) {
    pixel_format = PIXEL_FORMAT_NV21;
  } else if (img_format == ImageFormat::YUV420P_UV) {
    LOGE("YUV420P_UV not support, imageFormat: %d", (int32_t)img_format);
  } else if (img_format == ImageFormat::YUV420P_VU) {
    LOGE("YUV420P_VU not support, imageFormat: %d", (int32_t)img_format);
  } else if (img_format == ImageFormat::RGB_PACKED) {
    pixel_format = PIXEL_FORMAT_RGB_888;
  } else if (img_format == ImageFormat::BGR_PACKED) {
    pixel_format = PIXEL_FORMAT_BGR_888;
  } else if (img_format == ImageFormat::RGB_PLANAR) {
    pixel_format = PIXEL_FORMAT_RGB_888_PLANAR;
  } else if (img_format == ImageFormat::BGR_PLANAR) {
    pixel_format = PIXEL_FORMAT_BGR_888_PLANAR;
  } else {
    LOGE("imageFormat not support, imageFormat: %d", (int32_t)img_format);
    pixel_format = PIXEL_FORMAT_MAX;
  }

  return pixel_format;
}

#ifdef __CV184X__
static int32_t TDL_PLAT_VI_INIT(SAMPLE_VI_CONFIG_S *pstViConfig) {
  CVI_S32 s32Ret = CVI_SUCCESS;
  CVI_S32 i = 0;

  /************************************************
   * Set sns reset, probe; Set MIPI attr
   ************************************************/
  SAMPLE_COMM_VI_StartMIPI(pstViConfig);

  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    if (!pstViConfig->astViInfo->stDevInfo.bPatgen) {
      if (CVI_SNS_SetSnsProbe(i) != CVI_SUCCESS) {
        LOGE("[ERROR] sensor_%d probe failed!\n", i);
        return CVI_FAILURE;
      }
    }
  }

  /************************************************
   * Set VI dev config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartDev(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartDev failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }
  /************************************************
   * Set VI pipe config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartPipe(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartPipe failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }

  /************************************************
   * Create ISP
   ************************************************/
  // to do
  s32Ret = SAMPLE_COMM_VI_CreateIsp(pstViConfig);
  if (s32Ret != CVI_SUCCESS) {
    LOGE("[ERROR] SAMPLE_COMM_VI_CreateIsp failed with %#x!\n", s32Ret);
    return s32Ret;
  }
  /************************************************
   * Set sensor init
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    if (!pstViConfig->astViInfo->stDevInfo.bPatgen) {
      if (CVI_SNS_SetSnsInit(i) != CVI_SUCCESS) {
        LOGE("[ERROR] sensor_%d init failed!\n", i);
        return CVI_FAILURE;
      }
    }
  }
  /************************************************
   * Set VI chn config
   ************************************************/
  for (i = 0; i < pstViConfig->s32ViNum; i++) {
    s32Ret = SAMPLE_COMM_VI_StartChn(&pstViConfig->astViInfo[i]);
    if (s32Ret != CVI_SUCCESS) {
      LOGE("[ERROR] SAMPLE_COMM_VI_StartChn failed with %#x!\n", s32Ret);
      return s32Ret;
    }
  }

  return s32Ret;
}
#endif

int32_t ViDecoder::initialize(int32_t w, int32_t h, ImageFormat image_fmt,
                              int32_t vb_buffer_num) {
  if (isInitialized) {
    LOGI("Camera have isInitialized\n");
    return 0;
  }
  int32_t ret = 0;

#ifdef __CV184X__
  SNS_INI_CFG_S stIniCfg;
#else
  SAMPLE_INI_CFG_S stIniCfg;
#endif

  int32_t ViNum = 0;
  SAMPLE_VI_CONFIG_S stViConfig = {};
  VB_CONFIG_S stVbConf = {};
  PIC_SIZE_E enPicSize = {};
  VPSS_GRP VpssGrp = -1;
  int32_t pool_id[VI_MAX_PIPE_NUM] = {};
  PIXEL_FORMAT_E pix_fmt = convertPixelFormat(image_fmt);

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
    LOGE("memory_pool is nullptr");
    return -1;
  }

  VI_VPSS_MODE_S stVIVPSSMode;
  for (int i = 0; i < VI_MAX_PIPE_NUM; ++i) {
    stVIVPSSMode.aenMode[i] = VI_OFFLINE_VPSS_ONLINE;
  }
  CVI_SYS_SetVIVPSSMode(&stVIVPSSMode);

#ifndef __CV186X__

  VPSS_MODE_S stVPSSMode = {.enMode = VPSS_MODE_DUAL,
                            .aenInput = {VPSS_INPUT_MEM, VPSS_INPUT_ISP},
#ifndef __CV184X__
                            .ViPipe = {0}};
  CVI_SYS_SetVPSSModeEx(&stVPSSMode);
#else
                           };
  CVI_VPSS_SetMode(&stVPSSMode);
#endif

#endif

#ifdef __CV184X__
  ViNum = stViConfig.s32ViNum;
#else
  ViNum = stViConfig.s32WorkingViNum;
#endif

  SIZE_S stSize[ViNum] = {};
  for (int32_t i = 0; i < ViNum; i++) {
    ret = SAMPLE_COMM_VI_GetSizeBySensor(stIniCfg.enSnsType[i], &enPicSize);
    if (ret != CVI_SUCCESS) {
      LOGE("SAMPLE_COMM_VI_GetSizeBySensor failed with %#x\n", ret);
      return ret;
    }

    ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize[i]);
    if (ret != CVI_SUCCESS) {
      LOGE("SAMPLE_COMM_SYS_GetPicSize failed with %#x\n", ret);
      return ret;
    }
    memory_blocks_.push_back(
        pool->CreateExVb(vb_buffer_num, w, h, (void *)&pix_fmt));
    pool_id[i] = memory_blocks_.back()->id;
  }

  /************************************************
   * step3:  Init vi modules
   ************************************************/
#ifdef __CV184X__
  ret = TDL_PLAT_VI_INIT(&stViConfig);
  if (ret != 0) {
    LOGE("TDL_PLAT_VI_INIT failed. ret: 0x%x !\n", ret);
    return ret;
  }
#else
  ret = SAMPLE_PLAT_VI_INIT(&stViConfig);
  if (ret != 0) {
    LOGE("SAMPLE_PLAT_VI_INIT failed. ret: 0x%x !\n", ret);
    return ret;
  }
#endif

  /************************************************
   * step4:  Init vpss modules
   ************************************************/
  for (int32_t i = 0; i < ViNum; i++) {
    VpssGrp = i;

    VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
    VPSS_CHN VpssChn = VPSS_CHN0;
    VPSS_CHN_ATTR_S astVpssChnAttr = {0};
    CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {CVI_FALSE};

    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_NV21;
    stVpssGrpAttr.u32MaxW = stSize[i].u32Width;
    stVpssGrpAttr.u32MaxH = stSize[i].u32Height;
#ifndef __CV186X__
    stVpssGrpAttr.u8VpssDev = 1;
#endif
    astVpssChnAttr.u32Width = w;
    astVpssChnAttr.u32Height = h;
    astVpssChnAttr.enVideoFormat = VIDEO_FORMAT_LINEAR;
    astVpssChnAttr.enPixelFormat = pix_fmt;
    astVpssChnAttr.stFrameRate.s32SrcFrameRate = -1;
    astVpssChnAttr.stFrameRate.s32DstFrameRate = -1;
    astVpssChnAttr.u32Depth = 1;
    astVpssChnAttr.bMirror = CVI_FALSE;
    astVpssChnAttr.bFlip = CVI_FALSE;
    astVpssChnAttr.stAspectRatio.enMode = ASPECT_RATIO_NONE;
    astVpssChnAttr.stNormalize.bEnable = CVI_FALSE;

    abChnEnable[0] = CVI_TRUE;
#ifdef __CV184X__
    ret = SAMPLE_COMM_VPSS_INIT(VpssGrp, abChnEnable, &stVpssGrpAttr,
                                &astVpssChnAttr);
#else
    ret = SAMPLE_COMM_VPSS_Init(VpssGrp, abChnEnable, &stVpssGrpAttr,
                                &astVpssChnAttr);
#endif
    if (ret != CVI_SUCCESS) {
      LOGE("SAMPLE_COMM_VPSS_Init falied with ret: 0x%x !\n", ret);
      return ret;
    }

    ret = CVI_VPSS_AttachVbPool(VpssGrp, VpssChn, pool_id[i]);
    if (ret != CVI_SUCCESS) {
      LOGE("CVI_VPSS_AttachVbPool falied ret with 0x%x !\n", ret);
      return ret;
    }

    ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr,
                                 &astVpssChnAttr);
    if (ret != CVI_SUCCESS) {
      LOGE("start vpss group failed. ret: 0x%x !\n", ret);
      return ret;
    }

    ret = SAMPLE_COMM_VI_Bind_VPSS(i, i, VpssGrp);
    if (ret != CVI_SUCCESS) {
      LOGE("vi bind vpss failed. ret: 0x%x !\n", ret);
      return ret;
    }
  }

  isInitialized = true;

  return ret;
}

int32_t ViDecoder::deinitialize() {
  int32_t ret = 0;
  int32_t ViNum = 0;
  CVI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {CVI_FALSE};
  abChnEnable[0] = CVI_TRUE;

  for (int32_t i = 0; i < VI_MAX_PIPE_NUM; i++) {
    std::unique_lock<std::mutex> lock(queueMutexes[i]);
    while (!frameQueues[i].empty()) {
      auto frame_info = frameQueues[i].front();
      frameQueues[i].pop();
      for (int32_t j = 0; j < 3; j++) {
        if (frame_info->stVFrame.u32Length[j] != 0) {
          CVI_SYS_Munmap((void *)frame_info->stVFrame.pu8VirAddr[j],
                         frame_info->stVFrame.u32Length[j]);
        }
      }
      CVI_VPSS_ReleaseChnFrame(i, 0, frame_info.get());
    }
  }

  auto pool = std::dynamic_pointer_cast<CviMemoryPool>(memory_pool_);

#ifdef __CV184X__
  ViNum = g_stViConfig.s32ViNum;
#else
  ViNum = g_stViConfig.s32WorkingViNum;
#endif

  for (int32_t i = 0; i < ViNum; i++) {
    SAMPLE_COMM_VI_UnBind_VPSS(i, i, i);
    CVI_VPSS_DetachVbPool(i, 0);
    SAMPLE_COMM_VPSS_Stop(i, abChnEnable);
  }

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
  // if (!isInitialized) {
  //   initialize();
  // }
}

ViDecoder::~ViDecoder() {
  if (isInitialized) {
    deinitialize();
  }
}

int32_t ViDecoder::init(const std::string &path,
                        const std::map<std::string, int32_t> &config) {
  path_ = path;

  return 0;
}

int32_t ViDecoder::read(std::shared_ptr<BaseImage> &image, int32_t vi_chn) {
  int32_t ret = 0;
  std::shared_ptr<VIDEO_FRAME_INFO_S> frame_info =
      std::make_shared<VIDEO_FRAME_INFO_S>();
  while (true) {
    ret = CVI_VPSS_GetChnFrame(vi_chn, 0, frame_info.get(), 3000);
    VIDEO_FRAME_INFO_S *vpss_frame_info = frame_info.get();
    if (ret != 0 ||
        vpss_frame_info->stVFrame.u32Width == 0) {  // CVI_VPSS_GetChnFrame bug
      LOGE("CVI_VPSS_GetChnFrame(%d) failed, ret(%x) width %d\n", vi_chn, ret,
           vpss_frame_info->stVFrame.u32Width);
      return ret;
    } else {
      break;
    }
  }

  // 如果虚拟地址为空，进行内存映射
  if (frame_info->stVFrame.pu8VirAddr[0] == NULL) {
    isMapped_ = true;
    for (int32_t i = 0; i < 3; i++) {
      if (frame_info->stVFrame.u32Length[i] != 0) {
        frame_info->stVFrame.pu8VirAddr[i] =
            (CVI_U8 *)CVI_SYS_Mmap(frame_info->stVFrame.u64PhyAddr[i],
                                   frame_info->stVFrame.u32Length[i]);
        CVI_SYS_IonFlushCache(frame_info->stVFrame.u64PhyAddr[i],
                              frame_info->stVFrame.pu8VirAddr[i],
                              frame_info->stVFrame.u32Length[i]);
        addr_[i] = frame_info->stVFrame.pu8VirAddr[i];
        image_length_[i] = frame_info->stVFrame.u32Length[i];
      }
    }
  }

  image = ImageFactory::wrapVPSSFrame(frame_info.get(), false);

  std::lock_guard<std::mutex> lock(queueMutexes[vi_chn]);
  frameQueues[vi_chn].push(frame_info);

  return image ? 0 : -1;
}

int32_t ViDecoder::release(int32_t vi_chn) {
  int32_t ret = 0;
  if (frameQueues[vi_chn].empty()) {
    LOGE("FrameBuffer is empty\n");
    return -1;
  }

  std::unique_lock<std::mutex> lock(queueMutexes[vi_chn]);
  auto frame_info = frameQueues[vi_chn].front();
  frameQueues[vi_chn].pop();

  for (int32_t i = 0; i < 3; i++) {
    if (frame_info->stVFrame.u32Length[i] != 0) {
      CVI_SYS_Munmap((void *)frame_info->stVFrame.pu8VirAddr[i],
                     frame_info->stVFrame.u32Length[i]);
    }
  }

  ret = CVI_VPSS_ReleaseChnFrame(vi_chn, 0, frame_info.get());
  if (ret != 0) {
    LOGE("CVI_VPSS_ReleaseChnFrame(%d) failed with %d\n", vi_chn, ret);
  }
  return ret;
}
