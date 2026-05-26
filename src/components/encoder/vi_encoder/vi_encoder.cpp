#include "encoder/vi_encoder/vi_encoder.hpp"
#include "image/base_image.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

ViEncoder::ViEncoder(int VeChn, int width, int height, int fps, int bitrate,
                     int gop)
    : width_(width), height_(height), fps_(fps), bitrate_(bitrate), gop_(gop) {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  VeChn_ = VeChn;
  is_initialized_ = false;
  is_started_ = false;

  VENC_CHN_ATTR_S stAttr;
  memset(&stAttr, 0, sizeof(VENC_CHN_ATTR_S));

  if (CVI_SUCCESS == CVI_VENC_GetChnAttr(VeChn_, &stAttr)) {
    printf("[ViEncoder] venc channel %d already initialized, reuse it\n",
           VeChn_);
    is_initialized_ = true;
    return;
  }

  // --- VENC attr ---
  stAttr.stVencAttr.enType = PT_H264;
  stAttr.stVencAttr.u32MaxPicWidth = width_;
  stAttr.stVencAttr.u32MaxPicHeight = height_;
  stAttr.stVencAttr.u32PicWidth = width_;
  stAttr.stVencAttr.u32PicHeight = height_;
  stAttr.stVencAttr.u32Profile = H264E_PROFILE_MAIN;
  stAttr.stVencAttr.bByFrame = CVI_TRUE;
  stAttr.stVencAttr.bEsBufQueueEn = CVI_TRUE;
  stAttr.stVencAttr.bIsoSendFrmEn = CVI_TRUE;
  stAttr.stVencAttr.bSingleCore = CVI_FALSE;
  stAttr.stVencAttr.u32BufSize = width_ * height_ / 2;
  stAttr.stVencAttr.stAttrH264e.bRcnRefShareBuf = CVI_FALSE;
  stAttr.stVencAttr.stAttrH264e.bSingleLumaBuf = CVI_TRUE;

  // --- GOP attr ---
  stAttr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
  stAttr.stGopAttr.stNormalP.s32IPQpDelta = -2;

  // --- RC attr ---
  stAttr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
  stAttr.stRcAttr.stH264Cbr.u32Gop = gop_;
  stAttr.stRcAttr.stH264Cbr.u32StatTime = 1;
  stAttr.stRcAttr.stH264Cbr.u32SrcFrameRate = fps_;
  stAttr.stRcAttr.stH264Cbr.fr32DstFrameRate = fps_;
  stAttr.stRcAttr.stH264Cbr.u32BitRate = bitrate_;
  stAttr.stRcAttr.stH264Cbr.bVariFpsEn = 0;

  CVI_S32 s32Ret = CVI_VENC_CreateChn(VeChn_, &stAttr);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_CreateChn failed, ret=0x%x\n", s32Ret);
    return;
  }

  // --- ModParam ---
  VENC_PARAM_MOD_S stModParam;
  memset(&stModParam, 0, sizeof(VENC_PARAM_MOD_S));
  stModParam.enVencModType = MODTYPE_H264E;
  s32Ret = CVI_VENC_GetModParam(&stModParam);
  if (s32Ret == CVI_SUCCESS) {
    s32Ret = CVI_VENC_SetModParam(&stModParam);
    if (s32Ret != CVI_SUCCESS) {
      printf("[ViEncoder] CVI_VENC_SetModParam failed, ret=0x%x\n", s32Ret);
    }
  } else {
    printf("[ViEncoder] CVI_VENC_GetModParam failed, ret=0x%x\n", s32Ret);
  }

  // --- Entropy ---
  setH264Entropy();

  // --- H264 Trans ---
  setH264Trans();

  // --- VUI ---
  setH264Vui();

  // --- RC Param ---
  VENC_RC_PARAM_S stRcParam;
  s32Ret = CVI_VENC_GetRcParam(VeChn_, &stRcParam);
  if (s32Ret == CVI_SUCCESS) {
    stRcParam.u32ThrdLv = 2;
    stRcParam.s32InitialDelay = 1000;
    stRcParam.stParamH264Cbr.bQpMapEn = CVI_FALSE;
    stRcParam.stParamH264Cbr.s32MaxReEncodeTimes = 0;
    stRcParam.s32FirstFrameStartQp = 32;
    stRcParam.stParamH264Cbr.u32MaxIQp = 48;
    stRcParam.stParamH264Cbr.u32MinIQp = 12;
    stRcParam.stParamH264Cbr.u32MaxQp = 48;
    stRcParam.stParamH264Cbr.u32MinQp = 12;
    stRcParam.stParamH264Cbr.u32MaxIprop = 100;
    stRcParam.stParamH264Cbr.u32MinIprop = 1;
    s32Ret = CVI_VENC_SetRcParam(VeChn_, &stRcParam);
    if (s32Ret != CVI_SUCCESS) {
      printf("[ViEncoder] CVI_VENC_SetRcParam failed, ret=0x%x\n", s32Ret);
    }
  } else {
    printf("[ViEncoder] CVI_VENC_GetRcParam failed, ret=0x%x\n", s32Ret);
  }

  // StartRecvFrame is deferred to start() — matching app_ipcam_Venc_Init /
  // Start split

  is_initialized_ = true;
  printf("[ViEncoder] H264 channel %d created: %dx%d@%dfps, %dkbps, gop=%d\n",
         VeChn_, width_, height_, fps_, bitrate_, gop_);
#endif
}

ViEncoder::~ViEncoder() {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  if (is_initialized_) {
    if (is_started_) {
      CVI_VENC_StopRecvFrame(VeChn_);
    }
    CVI_VENC_ResetChn(VeChn_);
    CVI_VENC_DestroyChn(VeChn_);
    printf("[ViEncoder] H264 channel %d destroyed\n", VeChn_);
  }
#endif
}

bool ViEncoder::encodeFrame(const std::shared_ptr<BaseImage>& image,
                            std::vector<uint8_t>& encode_stream) {
  if (!image) {
    std::cerr << "[ViEncoder] Error: input image is nullptr.\n";
    return false;
  }

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  if (!is_initialized_) {
    std::cerr << "[ViEncoder] Error: encoder not initialized.\n";
    return false;
  }

  if (!is_started_) {
    if (start() != 0) {
      std::cerr << "[ViEncoder] Error: start() failed.\n";
      return false;
    }
  }

  if (image->getImageFormat() != ImageFormat::YUV420SP_VU &&
      image->getImageFormat() != ImageFormat::YUV420SP_UV) {
    std::cerr << "[ViEncoder] Error: image format must be YUV420SP.\n";
    return false;
  }

  VIDEO_FRAME_INFO_S* src_frame =
      static_cast<VIDEO_FRAME_INFO_S*>(image->getInternalData());
  if (!src_frame) {
    std::cerr << "[ViEncoder] Error: getInternalData() returned nullptr.\n";
    return false;
  }

  VENC_CHN_ATTR_S stAttr;
  CVI_VENC_GetChnAttr(VeChn_, &stAttr);
  if (stAttr.stVencAttr.u32PicWidth != src_frame->stVFrame.u32Width ||
      stAttr.stVencAttr.u32PicHeight != src_frame->stVFrame.u32Height) {
    stAttr.stVencAttr.u32PicWidth = src_frame->stVFrame.u32Width;
    stAttr.stVencAttr.u32PicHeight = src_frame->stVFrame.u32Height;
    CVI_VENC_SetChnAttr(VeChn_, &stAttr);
  }

  CVI_S32 s32Ret = CVI_VENC_SendFrame(VeChn_, src_frame, 2000);
  if (s32Ret != CVI_SUCCESS) {
    std::cerr << "[ViEncoder] CVI_VENC_SendFrame failed, ret=0x" << std::hex
              << s32Ret << std::dec << std::endl;
    return false;
  }

  VENC_CHN_STATUS_S stStatus;
  s32Ret = CVI_VENC_QueryStatus(VeChn_, &stStatus);
  if (s32Ret != CVI_SUCCESS) {
    std::cerr << "[ViEncoder] CVI_VENC_QueryStatus failed, ret=0x" << std::hex
              << s32Ret << std::dec << std::endl;
    return false;
  }

  VENC_STREAM_S stStream;
  memset(&stStream, 0, sizeof(VENC_STREAM_S));
  stStream.pstPack =
      (VENC_PACK_S*)malloc(sizeof(VENC_PACK_S) * stStatus.u32CurPacks);
  if (!stStream.pstPack) {
    std::cerr << "[ViEncoder] Error: malloc pack fail\n";
    return false;
  }

  s32Ret = CVI_VENC_GetStream(VeChn_, &stStream, 2000);
  if (s32Ret != CVI_SUCCESS || stStream.u32PackCount == 0) {
    std::cerr << "[ViEncoder] CVI_VENC_GetStream failed, ret=0x" << std::hex
              << s32Ret << " packCount=" << std::dec << stStream.u32PackCount
              << std::endl;
    free(stStream.pstPack);
    stStream.pstPack = NULL;
    return false;
  }

  uint32_t total_len = 0;
  for (uint32_t i = 0; i < stStream.u32PackCount; i++) {
    total_len += (stStream.pstPack[i].u32Len - stStream.pstPack[i].u32Offset);
  }

  encode_stream.resize(total_len);
  uint32_t offset = 0;
  for (uint32_t i = 0; i < stStream.u32PackCount; i++) {
    VENC_PACK_S* pstPack = &stStream.pstPack[i];
    uint32_t pack_len = pstPack->u32Len - pstPack->u32Offset;
    memcpy(encode_stream.data() + offset, pstPack->pu8Addr + pstPack->u32Offset,
           pack_len);
    offset += pack_len;
  }

  CVI_VENC_ReleaseStream(VeChn_, &stStream);
  if (stStream.pstPack != NULL) {
    free(stStream.pstPack);
    stStream.pstPack = NULL;
  }
  return true;

#else
  std::cerr << "[ViEncoder] Error: H264 hardware encoding not supported on "
               "this platform.\n";
  encode_stream.clear();
  return false;
#endif
}

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)

int32_t ViEncoder::setH264Entropy() {
  VENC_H264_ENTROPY_S h264Entropy;
  memset(&h264Entropy, 0, sizeof(VENC_H264_ENTROPY_S));
  h264Entropy.u32EntropyEncModeI = H264E_ENTROPY_CABAC;
  h264Entropy.u32EntropyEncModeP = H264E_ENTROPY_CABAC;

  CVI_S32 s32Ret = CVI_VENC_SetH264Entropy(VeChn_, &h264Entropy);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_SetH264Entropy failed, ret=0x%x\n", s32Ret);
    return s32Ret;
  }
  return 0;
}

int32_t ViEncoder::setH264Trans() {
  VENC_H264_TRANS_S h264Trans;
  memset(&h264Trans, 0, sizeof(VENC_H264_TRANS_S));

  CVI_S32 s32Ret = CVI_VENC_GetH264Trans(VeChn_, &h264Trans);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_GetH264Trans failed, ret=0x%x\n", s32Ret);
    return s32Ret;
  }

  h264Trans.chroma_qp_index_offset = 0;

  s32Ret = CVI_VENC_SetH264Trans(VeChn_, &h264Trans);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_SetH264Trans failed, ret=0x%x\n", s32Ret);
    return s32Ret;
  }
  return 0;
}

int32_t ViEncoder::setH264Vui() {
  VENC_H264_VUI_S h264Vui;
  memset(&h264Vui, 0, sizeof(VENC_H264_VUI_S));

  CVI_VENC_GetH264Vui(VeChn_, &h264Vui);

  h264Vui.stVuiTimeInfo.timing_info_present_flag = 1;
  h264Vui.stVuiTimeInfo.num_units_in_tick = 1;
  h264Vui.stVuiTimeInfo.time_scale = fps_;
  h264Vui.stVuiTimeInfo.fixed_frame_rate_flag = 1;

  h264Vui.stVuiAspectRatio.aspect_ratio_info_present_flag =
      CVI_H26X_ASPECT_RATIO_INFO_PRESENT_FLAG_DEFAULT;
  h264Vui.stVuiAspectRatio.overscan_info_present_flag =
      CVI_H26X_OVERSCAN_INFO_PRESENT_FLAG_DEFAULT;
  h264Vui.stVuiVideoSignal.video_signal_type_present_flag =
      CVI_H26X_VIDEO_SIGNAL_TYPE_PRESENT_FLAG_DEFAULT;

  CVI_S32 s32Ret = CVI_VENC_SetH264Vui(VeChn_, &h264Vui);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_SetH264Vui failed, ret=0x%x\n", s32Ret);
    return s32Ret;
  }
  return 0;
}

int32_t ViEncoder::start() {
  VENC_RECV_PIC_PARAM_S stRecvParam;
  stRecvParam.s32RecvPicNum = -1;
  CVI_S32 s32Ret = CVI_VENC_StartRecvFrame(VeChn_, &stRecvParam);
  if (s32Ret != CVI_SUCCESS) {
    printf("[ViEncoder] CVI_VENC_StartRecvFrame failed, ret=0x%x\n", s32Ret);
    return s32Ret;
  }
  is_started_ = true;
  printf("[ViEncoder] H264 channel %d started\n", VeChn_);
  return 0;
}
#endif