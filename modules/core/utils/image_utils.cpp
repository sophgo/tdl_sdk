#include "image_utils.hpp"

#include "core/cviai_utils.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "cviai_log.hpp"
#include "rescale_utils.hpp"

#define FACE_IMAGE_H 112
#define FACE_IMAGE_W 112

static void GET_BBOX_COORD(cvai_bbox_t *bbox, uint32_t &x1, uint32_t &y1, uint32_t &x2,
                           uint32_t &y2, uint32_t &height, uint32_t &width, PIXEL_FORMAT_E fmt,
                           uint32_t frame_height, uint32_t frame_width) {
  x1 = (uint32_t)floor(bbox->x1);
  y1 = (uint32_t)floor(bbox->y1);
  x2 = (uint32_t)floor(bbox->x2);
  y2 = (uint32_t)floor(bbox->y2);
  height = y2 - y1 + 1;
  width = x2 - x1 + 1;

  /* NOTE: tune the bbox coordinates to even value (necessary?) */
  switch (fmt) {
    case PIXEL_FORMAT_NV21: {
      if (height % 2 != 0) {
        if (y2 + 1 >= frame_height) {
          y1 -= 1;
        } else {
          y2 += 1;
        }
        height += 1;
      }
      if (width % 2 != 0) {
        if (x2 + 1 >= frame_width) {
          x1 -= 1;
        } else {
          x2 += 1;
        }
        width += 1;
      }
    } break;
    default:
      break;
  }
}

static void BBOX_PIXEL_COPY(uint8_t *src, uint8_t *dst, uint32_t stride_src, uint32_t stride_dst,
                            uint32_t x, uint32_t y, uint32_t w, uint32_t h, uint32_t bits) {
#if 0
  LOGI("[BBOX_PIXEL_COPY] src[%u], dst[%u], stride_src[%u], stride_dst[%u], x[%u], y[%u], w[%u], h[%u], bits[%u]\n",
         (uint32_t) src, (uint32_t) dst, stride_src, stride_dst, x, y, w, h, bits);
#endif
  for (uint32_t t = 0; t < h; t++) {
    memcpy(dst + t * stride_dst, src + (y + t) * stride_src + x * bits, w * bits);
  }
}

namespace cviai {

CVI_S32 crop_image(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image, cvai_bbox_t *bbox) {
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21) {
    LOGE("Pixel format [%d] not match PIXEL_FORMAT_RGB_888 [%d], PIXEL_FORMAT_NV21 [%d].\n",
         srcFrame->stVFrame.enPixelFormat, PIXEL_FORMAT_RGB_888, PIXEL_FORMAT_NV21);
    return CVI_FAILURE;
  }

  uint32_t x1, y1, x2, y2, height, width;
  GET_BBOX_COORD(bbox, x1, y1, x2, y2, height, width, srcFrame->stVFrame.enPixelFormat,
                 srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width);

  CVI_S32 ret = CVI_AI_CreateImage(dst_image, height, width, srcFrame->stVFrame.enPixelFormat);
  if (ret != CVIAI_SUCCESS) {
    return ret;
  }
  bool do_unmap = false;
  size_t frame_size = srcFrame->stVFrame.u32Length[0] + srcFrame->stVFrame.u32Length[1] +
                      srcFrame->stVFrame.u32Length[2];
  if (srcFrame->stVFrame.pu8VirAddr[0] == NULL) {
    srcFrame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0], frame_size);
    srcFrame->stVFrame.pu8VirAddr[1] =
        srcFrame->stVFrame.pu8VirAddr[0] + srcFrame->stVFrame.u32Length[0];
    srcFrame->stVFrame.pu8VirAddr[2] =
        srcFrame->stVFrame.pu8VirAddr[1] + srcFrame->stVFrame.u32Length[1];
    do_unmap = true;
  }

  switch (srcFrame->stVFrame.enPixelFormat) {
    case PIXEL_FORMAT_RGB_888: {
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[0], dst_image->pix[0],
                      srcFrame->stVFrame.u32Stride[0], dst_image->stride[0], x1, y1, width, height,
                      3);
    } break;
    case PIXEL_FORMAT_NV21: {
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[0], dst_image->pix[0],
                      srcFrame->stVFrame.u32Stride[0], dst_image->stride[0], x1, y1, width, height,
                      1);
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[1], dst_image->pix[1],
                      srcFrame->stVFrame.u32Stride[1], dst_image->stride[1], (x1 >> 1), (y1 >> 1),
                      (width >> 1), (height >> 1), 2);
    } break;
    default:
      break;
  }

  if (do_unmap) {
    CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
    srcFrame->stVFrame.pu8VirAddr[0] = NULL;
    srcFrame->stVFrame.pu8VirAddr[1] = NULL;
    srcFrame->stVFrame.pu8VirAddr[2] = NULL;
  }

  return CVIAI_SUCCESS;
}

static VIDEO_FRAME_INFO_S g_wrap_frame;

CVI_S32 crop_image_face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image,
                        cvai_face_info_t *face_info, bool align) {
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21) {
    LOGE("Pixel format [%d] not match PIXEL_FORMAT_RGB_888 [%d], PIXEL_FORMAT_NV21 [%d].\n",
         srcFrame->stVFrame.enPixelFormat, PIXEL_FORMAT_RGB_888, PIXEL_FORMAT_NV21);
    return CVIAI_ERR_INVALID_ARGS;
  }
  if (!align) {
    return crop_image(srcFrame, dst_image, &face_info->bbox);
  }
  // TODO: Check crop aligned face for NV21
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Crop aligned face only support PIXEL_FORMAT_RGB_888 now.\n");
    return CVIAI_ERR_INVALID_ARGS;
  }
  if (g_wrap_frame.stVFrame.u32Height == 0) {
    CREATE_ION_HELPER(&g_wrap_frame, FACE_IMAGE_W, FACE_IMAGE_H, PIXEL_FORMAT_RGB_888, "tpu");
  }
  CVI_AI_CreateImage(dst_image, FACE_IMAGE_H, FACE_IMAGE_W, srcFrame->stVFrame.enPixelFormat);

  cvai_face_info_t face_info_rescale =
      info_rescale_c(srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height,
                     srcFrame->stVFrame.u32Width, srcFrame->stVFrame.u32Height, *face_info);
  bool do_unmap = false;
  if (srcFrame->stVFrame.pu8VirAddr[0] == NULL) {
    srcFrame->stVFrame.pu8VirAddr[0] = (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0],
                                                                   srcFrame->stVFrame.u32Length[0]);
    do_unmap = true;
  }
  cv::Mat image(srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width, CV_8UC3,
                srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Stride[0]);
#if 0 /* TODO: why this don't work? (Segment fault) */
  cv::Mat warp_image(cv::Size(dst_image->width, dst_image->height), image.type(), dst_image->pix, dst_image->stride);
#endif
  cv::Mat warp_image(cv::Size(g_wrap_frame.stVFrame.u32Width, g_wrap_frame.stVFrame.u32Height),
                     image.type(), g_wrap_frame.stVFrame.pu8VirAddr[0],
                     g_wrap_frame.stVFrame.u32Stride[0]);

  if (face_align(image, warp_image, face_info_rescale) != 0) {
    return CVIAI_FAILURE;
  }

  BBOX_PIXEL_COPY(g_wrap_frame.stVFrame.pu8VirAddr[0], dst_image->pix[0],
                  g_wrap_frame.stVFrame.u32Stride[0], dst_image->stride[0], 0, 0, FACE_IMAGE_H,
                  FACE_IMAGE_H, 3);

  if (do_unmap) {
    CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
    srcFrame->stVFrame.pu8VirAddr[0] = NULL;
    srcFrame->stVFrame.pu8VirAddr[1] = NULL;
    srcFrame->stVFrame.pu8VirAddr[2] = NULL;
  }

#if 0 /* for debug */
  cv::cvtColor(warp_image, warp_image, cv::COLOR_RGB2BGR);
  cv::imwrite("visual/aligned_face.jpg", warp_image);
#endif

  return CVIAI_SUCCESS;
}

}  // namespace cviai