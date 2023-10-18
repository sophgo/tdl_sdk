#include "evaluation/cviai_media.h"
#include "cviai_log.hpp"

#include <iostream>
#include "core/core/cvai_errno.h"
#include "core/utils/vpss_helper.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// TODO: use memcpy
inline void BufferRGBPackedCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[2];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[0];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        uint32_t offset = i * 3 + j * vFrame->u32Stride[0];
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][offset] = ptr_pxl[0];
        vFrame->pu8VirAddr[0][offset + 1] = ptr_pxl[1];
        vFrame->pu8VirAddr[0][offset + 2] = ptr_pxl[2];
      }
    }
  }
}

inline void BufferRGBPacked2PlanarCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                       uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  if (invert) {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[2];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[0];
      }
    }
  } else {
    for (uint32_t j = 0; j < height; j++) {
      const uint8_t *ptr = buffer + j * stride;
      for (uint32_t i = 0; i < width; i++) {
        const uint8_t *ptr_pxl = i * 3 + ptr;
        vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[0];
        vFrame->pu8VirAddr[1][i + j * vFrame->u32Stride[1]] = ptr_pxl[1];
        vFrame->pu8VirAddr[2][i + j * vFrame->u32Stride[2]] = ptr_pxl[2];
      }
    }
  }
}
// input is bgr format
inline void BufferRGBPacked2YUVPlanarCopy(const uint8_t *buffer, uint32_t width, uint32_t height,
                                          uint32_t stride, VIDEO_FRAME_INFO_S *frame, bool invert) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  CVI_U8 *pY = vFrame->pu8VirAddr[0];
  CVI_U8 *pU = vFrame->pu8VirAddr[1];
  CVI_U8 *pV = vFrame->pu8VirAddr[2];

  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    for (uint32_t i = 0; i < width; i++) {
      const uint8_t *ptr_pxl = i * 3 + ptr;
      int b = ptr_pxl[0];
      int g = ptr_pxl[1];
      int r = ptr_pxl[2];
      if (invert) {
        std::swap(b, r);
      }
      pY[i + j * vFrame->u32Stride[0]] = ((66 * r + 129 * g + 25 * b) >> 8) + 16;

      if (j % 2 == 0 && i % 2 == 0) {
        pU[width / 2 + height / 2 * vFrame->u32Stride[1]] =
            ((-38 * r - 74 * g + 112 * b) >> 8) + 128;
        pV[width / 2 + height / 2 * vFrame->u32Stride[2]] =
            ((112 * r - 94 * g - 18 * b) >> 8) + 128;
      }
    }
  }
}
inline void BufferGreyCopy(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                           VIDEO_FRAME_INFO_S *frame) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    for (uint32_t i = 0; i < width; i++) {
      const uint8_t *ptr_pxl = ptr + i;
      vFrame->pu8VirAddr[0][i + j * vFrame->u32Stride[0]] = ptr_pxl[0];
    }
  }
}

template <typename T>
inline void BufferC12C1Copy(const uint8_t *buffer, uint32_t width, uint32_t height, uint32_t stride,
                            VIDEO_FRAME_INFO_S *frame) {
  VIDEO_FRAME_S *vFrame = &frame->stVFrame;
  for (uint32_t j = 0; j < height; j++) {
    const uint8_t *ptr = buffer + j * stride;
    uint8_t *vframec0ptr = vFrame->pu8VirAddr[0] + j * vFrame->u32Stride[0];
    memcpy(vframec0ptr, ptr, width * sizeof(T));
  }
}

CVI_S32 CVI_AI_Buffer2VBFrame(const uint8_t *buffer, uint32_t width, uint32_t height,
                              uint32_t stride, const PIXEL_FORMAT_E inFormat, VB_BLK *blk,
                              VIDEO_FRAME_INFO_S *frame, const PIXEL_FORMAT_E outFormat) {
  if (CREATE_VBFRAME_HELPER(blk, frame, width, height, outFormat) != CVI_SUCCESS) {
    LOGE("Create VBFrame failed.\n");
    return CVIAI_FAILURE;
  }

  int ret = CVIAI_SUCCESS;
  if ((inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_BGR_888) ||
      (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_RGB_888)) {
    BufferRGBPackedCopy(buffer, width, height, stride, frame, true);
  } else if ((inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_RGB_888) ||
             (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_BGR_888)) {
    BufferRGBPackedCopy(buffer, width, height, stride, frame, false);
  } else if (inFormat == PIXEL_FORMAT_BGR_888 && outFormat == PIXEL_FORMAT_RGB_888_PLANAR) {
    BufferRGBPacked2PlanarCopy(buffer, width, height, stride, frame, true);
  } else if (inFormat == PIXEL_FORMAT_RGB_888 && outFormat == PIXEL_FORMAT_RGB_888_PLANAR) {
    BufferRGBPacked2PlanarCopy(buffer, width, height, stride, frame, false);
  } else if (inFormat == PIXEL_FORMAT_BF16_C1 && outFormat == PIXEL_FORMAT_BF16_C1) {
    BufferC12C1Copy<uint16_t>(buffer, width, height, stride, frame);
  } else if (inFormat == PIXEL_FORMAT_FP32_C1 && outFormat == PIXEL_FORMAT_FP32_C1) {
    BufferC12C1Copy<float>(buffer, width, height, stride, frame);
  } else {
    LOGE("Unsupported convert format: %u -> %u.\n", inFormat, outFormat);
    ret = CVIAI_FAILURE;
  }
  CACHED_VBFRAME_FLUSH_UNMAP(frame);
  return ret;
}

CVI_S32 CVI_AI_ReadImage(const char *filepath, VIDEO_FRAME_INFO_S *frame, PIXEL_FORMAT_E format) {
  int ret = CVIAI_SUCCESS;
  try {
    cv::Mat img = cv::imread(filepath);
    if (img.empty()) {
      LOGE("Cannot read image %s.\n", filepath);
      return CVIAI_FAILURE;
    }

    if (CREATE_ION_HELPER(frame, img.cols, img.rows, format, "cviai/image") != CVIAI_SUCCESS) {
      LOGE("alloc ion failed, imgwidth:%d,imgheight:%d\n", img.cols, img.rows);
      return CVIAI_FAILURE;
    }
    switch (format) {
      case PIXEL_FORMAT_RGB_888: {
        BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, true);
      } break;
      case PIXEL_FORMAT_BGR_888: {
        BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, false);
      } break;
      case PIXEL_FORMAT_RGB_888_PLANAR: {
        BufferRGBPacked2PlanarCopy(img.data, img.cols, img.rows, img.step, frame, true);
      } break;
      case PIXEL_FORMAT_YUV_400: {
        cv::Mat img2;
        cv::cvtColor(img, img2, cv::COLOR_BGR2GRAY);
        BufferGreyCopy(img2.data, img2.cols, img2.rows, img2.step, frame);
      } break;
      case PIXEL_FORMAT_YUV_PLANAR_420:
        BufferRGBPacked2YUVPlanarCopy(img.data, img.cols, img.rows, img.step, frame, false);
        break;
      default:
        LOGE("Unsupported format: %u.\n", format);
        ret = CVIAI_FAILURE;
        break;
    }
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    std::cout << "exception caught: " << err_msg << std::endl;
    std::cout << "when read image: " << std::string(filepath) << std::endl;
    ret = CVIAI_FAILURE;
  }
  return ret;
}

CVI_S32 CVI_AI_ReadImage_Resize(const char *filepath, VIDEO_FRAME_INFO_S *frame,
                                PIXEL_FORMAT_E format, uint32_t width, uint32_t height) {
  int ret = CVIAI_SUCCESS;
  try {
    cv::Mat src_img = cv::imread(filepath);
    if (src_img.empty()) {
      LOGE("Cannot read image %s.\n", filepath);
      return CVIAI_FAILURE;
    }

    cv::Mat img;
    cv::resize(src_img, img, cv::Size(width, height));

    if (CREATE_ION_HELPER(frame, img.cols, img.rows, format, "cviai/image") != CVIAI_SUCCESS) {
      LOGE("alloc ion failed, imgwidth:%d,imgheight:%d\n", img.cols, img.rows);
      return CVIAI_FAILURE;
    }
    switch (format) {
      case PIXEL_FORMAT_RGB_888: {
        BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, true);
      } break;
      case PIXEL_FORMAT_BGR_888: {
        BufferRGBPackedCopy(img.data, img.cols, img.rows, img.step, frame, false);
      } break;
      case PIXEL_FORMAT_RGB_888_PLANAR: {
        BufferRGBPacked2PlanarCopy(img.data, img.cols, img.rows, img.step, frame, true);
      } break;
      case PIXEL_FORMAT_YUV_400: {
        cv::Mat img2;
        cv::cvtColor(img, img2, cv::COLOR_BGR2GRAY);
        BufferGreyCopy(img2.data, img2.cols, img2.rows, img2.step, frame);
      } break;
      case PIXEL_FORMAT_YUV_PLANAR_420:
        BufferRGBPacked2YUVPlanarCopy(img.data, img.cols, img.rows, img.step, frame, false);
        break;
      default:
        LOGE("Unsupported format: %u.\n", format);
        ret = CVIAI_FAILURE;
        break;
    }
  } catch (cv::Exception &e) {
    const char *err_msg = e.what();
    std::cout << "exception caught: " << err_msg << std::endl;
    std::cout << "when read image: " << std::string(filepath) << std::endl;
    ret = CVIAI_FAILURE;
  }
  return ret;
}

CVI_S32 CVI_AI_ReleaseImage(VIDEO_FRAME_INFO_S *frame) {
  CVI_S32 ret = CVI_SUCCESS;
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    ret = CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}
