#include "encoder/image_encoder/image_encoder.hpp"
#include "image/base_image.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

ImageEncoder::ImageEncoder(int VeChn) {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)

  VENC_RECV_PIC_PARAM_S stRecvParam;
  VENC_CHN_ATTR_S stAttr;

  VENC_JPEG_PARAM_S stJpegParam;
  VeChn_ = VeChn;

  if (CVI_SUCCESS == CVI_VENC_GetChnAttr(VeChn_, &stAttr)) {
    printf("venc channel %d have been init \n", VeChn_);
  }

  memset(&stAttr, 0, sizeof(VENC_CHN_ATTR_S));
  stAttr.stVencAttr.enType = PT_JPEG;        // payload类型
  stAttr.stVencAttr.u32MaxPicWidth = 1920;   // 影像最大编码宽度
  stAttr.stVencAttr.u32MaxPicHeight = 1080;  // 影像最大编码高度
  stAttr.stVencAttr.u32PicHeight = 64;
  // src_frame->stVFrame.u32Height;  // 编码影像宽度
  stAttr.stVencAttr.u32PicWidth = 64;        // 编码影像高度
  stAttr.stVencAttr.u32BufSize = 128 * 128;  // Encoded bitstream buffer大小
  stAttr.stVencAttr.u32Profile = H264E_PROFILE_BASELINE;  // 编码的等级
  stAttr.stVencAttr.bByFrame =
      CVI_TRUE;  // Encoded bitstream收集方式
                 // CVI_TRUE: 以帧为主；CVI_FALSE:以封包为主

  stAttr.stGopAttr.enGopMode = VENC_GOPMODE_NORMALP;
  stAttr.stGopAttr.stNormalP.s32IPQpDelta = 0;
  stAttr.stVencAttr.stAttrJpege.bSupportDCF = CVI_FALSE;
  stAttr.stVencAttr.stAttrJpege.enReceiveMode = VENC_PIC_RECEIVE_SINGLE;
  stAttr.stVencAttr.stAttrJpege.stMPFCfg.u8LargeThumbNailNum = 0;

  // set channel
  CVI_VENC_CreateChn(VeChn_, &stAttr);

  CVI_VENC_GetJpegParam(VeChn_, &stJpegParam);
  stJpegParam.u32Qfactor = 20;
  CVI_VENC_SetJpegParam(VeChn_, &stJpegParam);
  stRecvParam.s32RecvPicNum = -1;
  CVI_VENC_StartRecvFrame(VeChn_, &stRecvParam);

#endif
}

ImageEncoder::~ImageEncoder() {
#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)
  CVI_VENC_StopRecvFrame(VeChn_);
  CVI_VENC_ResetChn(VeChn_);
  CVI_VENC_DestroyChn(VeChn_);
#endif
}

bool ImageEncoder::encodeFrame(const std::shared_ptr<BaseImage>& image,
                               std::vector<uint8_t>& encode_img,
                               int jpeg_quality) {
  if (!image) {
    std::cerr << "[ImageEncoder] Error: input image is nullptr.\n";
    return false;
  }

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)

  if (image->getImageFormat() != ImageFormat::YUV420SP_VU) {
    std::cerr << "[ImageEncoder] Error: image format is not YUV420SP_VU.\n";
    return false;
  }

  VIDEO_FRAME_INFO_S* src_frame =
      static_cast<VIDEO_FRAME_INFO_S*>(image->getInternalData());

  if (!src_frame) {
    std::cerr << "getFrame() failed!\n";
    return false;
  }

  VENC_STREAM_S stStream;
  VENC_PACK_S* pstPack;
  VENC_CHN_ATTR_S stAttr;
  VENC_CHN VeChn_ = 1;

  // Retrieve and set encoding attributes
  CVI_VENC_GetChnAttr(VeChn_, &stAttr);
  stAttr.stVencAttr.u32PicHeight = src_frame->stVFrame.u32Height;
  stAttr.stVencAttr.u32PicWidth = src_frame->stVFrame.u32Width;
  if (src_frame->stVFrame.u32Height >= 1080) {
    stAttr.stVencAttr.u32BufSize = 1024 * 512;  // Encoded bitstream buffer大小
  } else {
    VENC_JPEG_PARAM_S stJpegParam, *pstJpegParam = &stJpegParam;
    CVI_S32 s32Ret = CVI_SUCCESS;

    s32Ret = CVI_VENC_GetJpegParam(VeChn_, pstJpegParam);
    if (s32Ret != CVI_SUCCESS) {
      return false;
    }
    pstJpegParam->u32Qfactor = 90;

    s32Ret = CVI_VENC_SetJpegParam(VeChn_, pstJpegParam);
    if (s32Ret != CVI_SUCCESS) {
      return false;
    }
  }
  CVI_VENC_SetChnAttr(VeChn_, &stAttr);
  // Send frame to hardware
  CVI_VENC_SendFrame(VeChn_, src_frame, 2000);
  // Allocate pack
  stStream.pstPack = (VENC_PACK_S*)malloc(sizeof(VENC_PACK_S) * 8);
  if (!stStream.pstPack) {
    std::cerr << "[ImageEncoder] Error: malloc pack fail\n";
    return false;
  }
  // Retrieve encoded stream
  int ret = CVI_VENC_GetStream(VeChn_, &stStream, 2000);
  if (ret != 0) {
    std::cerr << "[ImageEncoder] CVI_VENC_GetStream failed, ret=" << ret
              << std::endl;
    free(stStream.pstPack);
    stStream.pstPack = NULL;
    return false;
  }
  // Calculate total length
  uint32_t total_len = 0;
  for (uint32_t i = 0; i < stStream.u32PackCount; i++) {
    pstPack = &stStream.pstPack[i];
    total_len += (pstPack->u32Len - pstPack->u32Offset);
  }
  // Allocate output buffer
  encode_img.resize(total_len);
  // Copy JPEG data
  uint32_t offset = 0;
  for (uint32_t j = 0; j < stStream.u32PackCount; j++) {
    pstPack = &stStream.pstPack[j];
    uint32_t pack_len = pstPack->u32Len - pstPack->u32Offset;
    memcpy(encode_img.data() + offset, pstPack->pu8Addr + pstPack->u32Offset,
           pack_len);
    offset += pack_len;
  }
  // Release stream
  CVI_VENC_ReleaseStream(VeChn_, &stStream);
  if (stStream.pstPack != NULL) {
    free(stStream.pstPack);
    stStream.pstPack = NULL;
  }
  return true;

#else
  cv::Mat src = *(cv::Mat*)image->getInternalData();

  std::vector<int> param{cv::IMWRITE_JPEG_QUALITY, jpeg_quality};

  encode_img.clear();
  if (!cv::imencode(".jpg", src, encode_img, param)) {
    std::cerr << "[ImageEncoder] Error: cv::imencode failed.\n";
    return false;
  }

  return true;

#endif
}
