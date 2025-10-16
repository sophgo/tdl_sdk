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
                               std::vector<uint8_t>& encode_img, int VeChn,
                               int jpeg_quality) {
  if (!image) {
    std::cerr << "[ImageEncoder] Error: input image is nullptr.\n";
    return false;
  }

#if defined(__CV181X__) || defined(__CV180X__) || defined(__CV182X__) || \
    defined(__CV183X__) || defined(__CV184X__) || defined(__CV186X__)

  // 允许 NV21(VU) 与 NV12(UV)
  if (image->getImageFormat() != ImageFormat::YUV420SP_VU &&
      image->getImageFormat() != ImageFormat::YUV420SP_UV) {
    std::cerr << "[ImageEncoder] Error: image format is not YUV420SP_VU/UV.\n";
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
  VeChn_ = VeChn;

  // 获取并设置编码属性（宽高、缓冲等）
  CVI_VENC_GetChnAttr(VeChn_, &stAttr);
  stAttr.stVencAttr.u32PicHeight = src_frame->stVFrame.u32Height;
  stAttr.stVencAttr.u32PicWidth = src_frame->stVFrame.u32Width;

  if (src_frame->stVFrame.u32Height >= 1080) {
    stAttr.stVencAttr.u32BufSize = 1024 * 512;
  } else {
    VENC_JPEG_PARAM_S stJpegParam, *pstJpegParam = &stJpegParam;
    CVI_S32 s32Ret = CVI_VENC_GetJpegParam(VeChn_, pstJpegParam);
    if (s32Ret != CVI_SUCCESS) {
      return false;
    }
    pstJpegParam->u32Qfactor = jpeg_quality;  // 使用传入质量
    s32Ret = CVI_VENC_SetJpegParam(VeChn_, pstJpegParam);
    if (s32Ret != CVI_SUCCESS) {
      return false;
    }
  }

  // 如需区分 NV12/NV21，可在创建编码通道时设置 stAttr.stVencAttr.enPixelFormat
  // 这里只保留尺寸与质量的调整
  CVI_VENC_SetChnAttr(VeChn_, &stAttr);

  // 送帧到硬件
  CVI_VENC_SendFrame(VeChn_, src_frame, 2000);

  // 申请包
  stStream.pstPack = (VENC_PACK_S*)malloc(sizeof(VENC_PACK_S) * 8);
  if (!stStream.pstPack) {
    std::cerr << "[ImageEncoder] Error: malloc pack fail\n";
    return false;
  }

  // 取码流
  int ret = CVI_VENC_GetStream(VeChn_, &stStream, 2000);
  if (ret != 0) {
    std::cerr << "[ImageEncoder] CVI_VENC_GetStream failed, ret=" << ret
              << std::endl;
    free(stStream.pstPack);
    stStream.pstPack = NULL;
    return false;
  }

  // 计算总长度
  uint32_t total_len = 0;
  for (uint32_t i = 0; i < stStream.u32PackCount; i++) {
    pstPack = &stStream.pstPack[i];
    total_len += (pstPack->u32Len - pstPack->u32Offset);
  }

  // 拷贝输出
  encode_img.resize(total_len);
  uint32_t offset = 0;
  for (uint32_t j = 0; j < stStream.u32PackCount; j++) {
    pstPack = &stStream.pstPack[j];
    uint32_t pack_len = pstPack->u32Len - pstPack->u32Offset;
    memcpy(encode_img.data() + offset, pstPack->pu8Addr + pstPack->u32Offset,
           pack_len);
    offset += pack_len;
  }

  // 释放码流
  CVI_VENC_ReleaseStream(VeChn_, &stStream);
  if (stStream.pstPack != NULL) {
    free(stStream.pstPack);
    stStream.pstPack = NULL;
  }
  return true;

#else
  // 非芯片平台：支持 BGR/RGB，及 NV12/NV21
  encode_img.clear();

  auto fmt = image->getImageFormat();
  if (fmt == ImageFormat::BGR_PACKED || fmt == ImageFormat::RGB_PACKED) {
    cv::Mat src = *(cv::Mat*)image->getInternalData();
    std::vector<int> param{cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    if (!cv::imencode(".jpg", src, encode_img, param)) {
      std::cerr << "[ImageEncoder] Error: cv::imencode failed.\n";
      return false;
    }
    return true;
  }

  if (fmt != ImageFormat::YUV420SP_VU && fmt != ImageFormat::YUV420SP_UV) {
    std::cerr
        << "[ImageEncoder] Error: unsupported image format on non-chip path.\n";
    return false;
  }

  // 将 Y/UV 两平面按 stride 拷贝为紧凑 NV 图，然后转为 BGR
  const uint32_t width = image->getWidth();
  const uint32_t height = image->getHeight();

  size_t nv_size = static_cast<size_t>(width) * height * 3 / 2;
  unsigned char* nv_data = (unsigned char*)malloc(nv_size);
  if (!nv_data) {
    std::cerr << "[ImageEncoder] Error: allocate NV buffer failed.\n";
    return false;
  }

  // 拷贝 Y
  {
    const uint8_t* src_y =
        reinterpret_cast<const uint8_t*>(image->getVirtualAddress()[0]);
    const int src_y_stride =
        image->getStrides()[0] > 0 ? image->getStrides()[0] : (int)width;
    uint8_t* dst_y = nv_data;
    for (uint32_t r = 0; r < height; ++r) {
      memcpy(dst_y + r * width, src_y + r * src_y_stride, width);
    }
  }

  // 拷贝 UV/VU
  {
    const uint8_t* src_uv =
        reinterpret_cast<const uint8_t*>(image->getVirtualAddress()[1]);
    const int src_uv_stride =
        (image->getStrides().size() > 1 && image->getStrides()[1] > 0)
            ? image->getStrides()[1]
            : (int)width;
    uint8_t* dst_uv = nv_data + width * height;
    const uint32_t uv_rows = height / 2;
    for (uint32_t r = 0; r < uv_rows; ++r) {
      memcpy(dst_uv + r * width, src_uv + r * src_uv_stride, width);
    }
  }

  cv::Mat nv_img(height + height / 2, width, CV_8UC1, nv_data);
  int cvt_code = (fmt == ImageFormat::YUV420SP_UV)
                     ? cv::COLOR_YUV2BGR_NV12   // NV12
                     : cv::COLOR_YUV2BGR_NV21;  // NV21
  cv::Mat bgr_img;
  cv::cvtColor(nv_img, bgr_img, cvt_code);

  std::vector<int> param{cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
  bool ok = cv::imencode(".jpg", bgr_img, encode_img, param);

  free(nv_data);

  if (!ok) {
    std::cerr << "[ImageEncoder] Error: cv::imencode failed.\n";
    return false;
  }
  return true;
#endif
}
