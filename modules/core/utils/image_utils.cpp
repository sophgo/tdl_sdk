#include "image_utils.hpp"

#include "core/cviai_utils.h"
#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "cviai_log.hpp"
#include "rescale_utils.hpp"

#include "opencv2/core.hpp"
#ifdef ENABLE_CVIAI_CV_UTILS
#include "cv/imgproc.hpp"
#else
#include "opencv2/imgproc.hpp"
#endif

#include <sys/time.h>

#define FACE_IMAGE_H 112
#define FACE_IMAGE_W 112
#define FACE_CROP_EXTEN_RATIO (0.2)

static bool IS_SUPPORTED_FORMAT(VIDEO_FRAME_INFO_S *frame) {
  if (frame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_NV21 &&
      frame->stVFrame.enPixelFormat != PIXEL_FORMAT_YUV_PLANAR_420) {
    LOGE(
        "Pixel format [%d] not match: , PIXEL_FORMAT_RGB_888 [%d], PIXEL_FORMAT_NV21 [%d]"
        ", PIXEL_FORMAT_YUV_PLANAR_420 [%d].\n",
        frame->stVFrame.enPixelFormat, PIXEL_FORMAT_RGB_888, PIXEL_FORMAT_NV21,
        PIXEL_FORMAT_YUV_PLANAR_420);
    return false;
  }
  return true;
}

static void DO_MAP_IF_NEEDED(VIDEO_FRAME_INFO_S *frame, bool *do_unmap) {
  *do_unmap = false;
  CVI_U32 frame_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (frame->stVFrame.pu8VirAddr[0] == NULL) {
    frame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(frame->stVFrame.u64PhyAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[1] = frame->stVFrame.pu8VirAddr[0] + frame->stVFrame.u32Length[0];
    frame->stVFrame.pu8VirAddr[2] = frame->stVFrame.pu8VirAddr[1] + frame->stVFrame.u32Length[1];
    *do_unmap = true;
  }
}

static void DO_UNMAP_IF_NEEDED(VIDEO_FRAME_INFO_S *frame, bool do_unmap) {
  CVI_U32 frame_size =
      frame->stVFrame.u32Length[0] + frame->stVFrame.u32Length[1] + frame->stVFrame.u32Length[2];
  if (do_unmap) {
    CVI_SYS_Munmap((void *)frame->stVFrame.pu8VirAddr[0], frame_size);
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
}

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
    case PIXEL_FORMAT_RGB_888:
    case PIXEL_FORMAT_YUV_PLANAR_420:
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

static void BBOX_PIXEL_COPY_2(uint8_t *src, uint8_t *dst, uint32_t src_width, uint32_t src_height,
                              uint32_t stride_src, uint32_t stride_dst, int x, int y, uint32_t w,
                              uint32_t h, uint32_t bits) {
#if 0
  LOGI(
      "[BBOX_PIXEL_COPY] src[0x%x], dst[0x%x], src_width[%u], src_height[%u], stride_src[%u], stride_dst[%u], x[%d], "
      "y[%d], w[%u], h[%u], bits[%u]\n", (uint32_t) src, (uint32_t) dst,
      src_width, src_height, stride_src, stride_dst, x, y, w, h, bits);
#endif

  uint32_t w_offset = (x < 0) ? -1. * x : 0;
  uint32_t copy_width = (x + w < src_width) ? w : src_width - x;
  for (uint32_t t = 0; t < h; t++) {
    if (y + (int)t < 0 || y + (int)t >= (int)src_height) {
      continue;
    }
    memcpy(dst + t * stride_dst + w_offset * bits,
           src + (y + t) * stride_src + (x + w_offset) * bits, (copy_width - w_offset) * bits);
  }
}

namespace cviai {

CVI_S32 crop_image(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image, cvai_bbox_t *bbox,
                   bool cvtRGB888) {
  if (false == IS_SUPPORTED_FORMAT(srcFrame)) {
    return CVI_FAILURE;
  }

  uint32_t x1, y1, x2, y2, height, width;
  GET_BBOX_COORD(bbox, x1, y1, x2, y2, height, width, srcFrame->stVFrame.enPixelFormat,
                 srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width);

  cvai_image_t tmp_image;
  memset(&tmp_image, 0, sizeof(cvai_image_t));
  if (cvtRGB888 && srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    if (CVIAI_SUCCESS !=
        CVI_AI_CreateImage(&tmp_image, height, width, srcFrame->stVFrame.enPixelFormat)) {
      return CVIAI_FAILURE;
    }
    if (CVIAI_SUCCESS != CVI_AI_CreateImage(dst_image, height, width, PIXEL_FORMAT_RGB_888)) {
      return CVIAI_FAILURE;
    }
  } else {
    if (CVIAI_SUCCESS !=
        CVI_AI_CreateImage(dst_image, height, width, srcFrame->stVFrame.enPixelFormat)) {
      return CVIAI_FAILURE;
    }
  }

  bool do_unmap = false;
  DO_MAP_IF_NEEDED(srcFrame, &do_unmap);

  switch (srcFrame->stVFrame.enPixelFormat) {
    case PIXEL_FORMAT_RGB_888: {
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[0], dst_image->pix[0],
                      srcFrame->stVFrame.u32Stride[0], dst_image->stride[0], x1, y1, width, height,
                      3);
    } break;
    case PIXEL_FORMAT_NV21: {
      cvai_image_t *p_image = (cvtRGB888) ? &tmp_image : dst_image;
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[0], p_image->pix[0],
                      srcFrame->stVFrame.u32Stride[0], p_image->stride[0], x1, y1, width, height,
                      1);
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[1], p_image->pix[1],
                      srcFrame->stVFrame.u32Stride[1], p_image->stride[1], (x1 >> 1), (y1 >> 1),
                      (width >> 1), (height >> 1), 2);
      if (cvtRGB888) {
        cv::Mat cvImage_NV21(tmp_image.height * 3 / 2, tmp_image.width, CV_8UC1, tmp_image.pix[0],
                             tmp_image.stride[0]);
        cv::Mat cvImage_RGB888(dst_image->height, dst_image->width, CV_8UC3, dst_image->pix[0],
                               dst_image->stride[0]);
#ifdef ENABLE_CVIAI_CV_UTILS
        cviai::cvtColor(cvImage_NV21, cvImage_RGB888, CV_YUV2RGB_NV21);
#else
        cv::cvtColor(cvImage_NV21, cvImage_RGB888, CV_YUV2RGB_NV21);
#endif
      }
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_420: {
      cvai_image_t *p_image = (cvtRGB888) ? &tmp_image : dst_image;
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[0], p_image->pix[0],
                      srcFrame->stVFrame.u32Stride[0], p_image->stride[0], x1, y1, width, height,
                      1);
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[1], p_image->pix[1],
                      srcFrame->stVFrame.u32Stride[1], p_image->stride[1], (x1 >> 1), (y1 >> 1),
                      (width >> 1), (height >> 1), 1);
      BBOX_PIXEL_COPY(srcFrame->stVFrame.pu8VirAddr[2], p_image->pix[2],
                      srcFrame->stVFrame.u32Stride[2], p_image->stride[2], (x1 >> 1), (y1 >> 1),
                      (width >> 1), (height >> 1), 1);
      if (cvtRGB888) {
        cv::Mat cvImage_YUV420(tmp_image.height * 3 / 2, tmp_image.width, CV_8UC1, tmp_image.pix[0],
                               tmp_image.stride[0]);
        cv::Mat cvImage_RGB888(dst_image->height, dst_image->width, CV_8UC3, dst_image->pix[0],
                               dst_image->stride[0]);
#ifdef ENABLE_CVIAI_CV_UTILS
        cviai::cvtColor(cvImage_YUV420, cvImage_RGB888, CV_YUV2RGB_I420);
#else
        cv::cvtColor(cvImage_YUV420, cvImage_RGB888, CV_YUV2RGB_I420);
#endif
      }
    }
    default:
      break;
  }

  CVI_AI_Free(&tmp_image);
  DO_UNMAP_IF_NEEDED(srcFrame, do_unmap);

  return CVI_SUCCESS;
}

CVI_S32 crop_image_exten(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image, cvai_bbox_t *bbox,
                         bool cvtRGB888, float exten_ratio, float *offset_x, float *offset_y) {
  if (false == IS_SUPPORTED_FORMAT(srcFrame)) {
    return CVI_FAILURE;
  }

  uint32_t x1, y1, x2, y2, height, width;
  GET_BBOX_COORD(bbox, x1, y1, x2, y2, height, width, srcFrame->stVFrame.enPixelFormat,
                 srcFrame->stVFrame.u32Height, srcFrame->stVFrame.u32Width);
  bbox->x1 = x1;
  bbox->y1 = y1;
  bbox->x2 = x2;
  bbox->y2 = y2;
  uint32_t edge = MAX(height, width);
  uint32_t edge_exten = (uint32_t)(edge * exten_ratio);
  // uint32_t exten_image_edge = edge + 2 * edge_exten;

  int ext_x1 = (int)x1 - (int)(edge - width) / 2 - edge_exten;
  int ext_y1 = (int)y1 - (int)(edge - height) / 2 - edge_exten;
  *offset_x = edge_exten + (edge - width) / 2;
  *offset_y = edge_exten + (edge - height) / 2;

  cvai_image_t tmp_image;
  memset(&tmp_image, 0, sizeof(cvai_image_t));
  if (cvtRGB888 && srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    if (CVIAI_SUCCESS != CVI_AI_CreateImage(&tmp_image, edge + 2 * edge_exten,
                                            edge + 2 * edge_exten,
                                            srcFrame->stVFrame.enPixelFormat)) {
      return CVIAI_FAILURE;
    }
    if (CVIAI_SUCCESS != CVI_AI_CreateImage(dst_image, edge + 2 * edge_exten, edge + 2 * edge_exten,
                                            PIXEL_FORMAT_RGB_888)) {
      return CVIAI_FAILURE;
    }
  } else {
    if (CVIAI_SUCCESS != CVI_AI_CreateImage(dst_image, edge + 2 * edge_exten, edge + 2 * edge_exten,
                                            srcFrame->stVFrame.enPixelFormat)) {
      return CVIAI_FAILURE;
    }
  }

  bool do_unmap = false;
  DO_MAP_IF_NEEDED(srcFrame, &do_unmap);

  switch (srcFrame->stVFrame.enPixelFormat) {
    case PIXEL_FORMAT_RGB_888: {
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[0], dst_image->pix[0],
                        (int)srcFrame->stVFrame.u32Width, (int)srcFrame->stVFrame.u32Height,
                        srcFrame->stVFrame.u32Stride[0], dst_image->stride[0], ext_x1, ext_y1,
                        dst_image->width, dst_image->height, 3);
    } break;
    case PIXEL_FORMAT_NV21: {
      cvai_image_t *p_image = (cvtRGB888) ? &tmp_image : dst_image;
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[0], p_image->pix[0],
                        (int)srcFrame->stVFrame.u32Width, (int)srcFrame->stVFrame.u32Height,
                        srcFrame->stVFrame.u32Stride[0], p_image->stride[0], ext_x1, ext_y1,
                        p_image->width, p_image->height, 1);
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[1], p_image->pix[1],
                        (int)srcFrame->stVFrame.u32Width / 2, (int)srcFrame->stVFrame.u32Height / 2,
                        srcFrame->stVFrame.u32Stride[1], p_image->stride[1], (ext_x1 / 2),
                        (ext_y1 / 2), (p_image->width / 2), (p_image->height / 2), 2);
      if (cvtRGB888) {
        cv::Mat cvImage_NV21(tmp_image.height * 3 / 2, tmp_image.width, CV_8UC1, tmp_image.pix[0],
                             tmp_image.stride[0]);
        cv::Mat cvImage_RGB888(dst_image->height, dst_image->width, CV_8UC3, dst_image->pix[0],
                               dst_image->stride[0]);
#ifdef ENABLE_CVIAI_CV_UTILS
        cviai::cvtColor(cvImage_NV21, cvImage_RGB888, CV_YUV2RGB_NV21);
#else
        cv::cvtColor(cvImage_NV21, cvImage_RGB888, CV_YUV2RGB_NV21);
#endif
      }
    } break;
    case PIXEL_FORMAT_YUV_PLANAR_420: {
      cvai_image_t *p_image = (cvtRGB888) ? &tmp_image : dst_image;
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[0], p_image->pix[0],
                        (int)srcFrame->stVFrame.u32Width, (int)srcFrame->stVFrame.u32Height,
                        srcFrame->stVFrame.u32Stride[0], p_image->stride[0], ext_x1, ext_y1,
                        p_image->width, p_image->height, 1);
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[1], p_image->pix[1],
                        (int)srcFrame->stVFrame.u32Width / 2, (int)srcFrame->stVFrame.u32Height / 2,
                        srcFrame->stVFrame.u32Stride[1], p_image->stride[1], (ext_x1 / 2),
                        (ext_y1 / 2), (p_image->width / 2), (p_image->height / 2), 1);
      BBOX_PIXEL_COPY_2(srcFrame->stVFrame.pu8VirAddr[2], p_image->pix[2],
                        (int)srcFrame->stVFrame.u32Width / 2, (int)srcFrame->stVFrame.u32Height / 2,
                        srcFrame->stVFrame.u32Stride[2], p_image->stride[2], (ext_x1 / 2),
                        (ext_y1 / 2), (p_image->width / 2), (p_image->height / 2), 1);
      if (cvtRGB888) {
        cv::Mat cvImage_YUV420(tmp_image.height * 3 / 2, tmp_image.width, CV_8UC1, tmp_image.pix[0],
                               tmp_image.stride[0]);
        cv::Mat cvImage_RGB888(dst_image->height, dst_image->width, CV_8UC3, dst_image->pix[0],
                               dst_image->stride[0]);
#ifdef ENABLE_CVIAI_CV_UTILS
        cviai::cvtColor(cvImage_YUV420, cvImage_RGB888, CV_YUV2RGB_I420);
#else
        cv::cvtColor(cvImage_YUV420, cvImage_RGB888, CV_YUV2RGB_I420);
#endif
      }
    } break;
    default:
      break;
  }

  CVI_AI_Free(&tmp_image);
  DO_UNMAP_IF_NEEDED(srcFrame, do_unmap);

  return CVIAI_SUCCESS;
}

static CVI_S32 PREPARE_FACE_ALIGNMENT_DATA(VIDEO_FRAME_INFO_S *frame, cvai_image_t *image,
                                           cvai_face_info_t *face_info,
                                           cvai_face_info_t *new_face_info, uint32_t *height,
                                           uint32_t *width, uint32_t *stride, uint8_t **data_ptr) {
  if (frame->stVFrame.enPixelFormat == PIXEL_FORMAT_RGB_888) {
    *height = frame->stVFrame.u32Height;
    *width = frame->stVFrame.u32Width;
    *stride = frame->stVFrame.u32Stride[0];
    *data_ptr = frame->stVFrame.pu8VirAddr[0];
    new_face_info->pts.size = 5;
    new_face_info->pts.x = (float *)malloc(5 * sizeof(float));
    new_face_info->pts.y = (float *)malloc(5 * sizeof(float));
    for (int i = 0; i < 5; i++) {
      new_face_info->pts.x[i] = face_info->pts.x[i];
      new_face_info->pts.y[i] = face_info->pts.y[i];
    }
  } else {
    float x_offset, y_offset;
    if (CVIAI_SUCCESS != crop_image_exten(frame, image, &face_info->bbox, true,
                                          FACE_CROP_EXTEN_RATIO, &x_offset, &y_offset)) {
      return CVIAI_FAILURE;
    }
    *height = image->height;
    *width = image->width;
    *stride = image->stride[0];
    *data_ptr = image->pix[0];
    new_face_info->pts.size = 5;
    new_face_info->pts.x = (float *)malloc(5 * sizeof(float));
    new_face_info->pts.y = (float *)malloc(5 * sizeof(float));
    for (int i = 0; i < 5; i++) {
      new_face_info->pts.x[i] = face_info->pts.x[i] - face_info->bbox.x1 + x_offset;
      new_face_info->pts.y[i] = face_info->pts.y[i] - face_info->bbox.y1 + y_offset;
    }
  }
  return CVIAI_SUCCESS;
}

CVI_S32 crop_image_face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image,
                        cvai_face_info_t *face_info, bool align, bool cvtRGB888) {
  if (false == IS_SUPPORTED_FORMAT(srcFrame)) {
    return CVIAI_ERR_INVALID_ARGS;
  }

  if (!align) {
    return crop_image(srcFrame, dst_image, &face_info->bbox, cvtRGB888);
  }

  bool do_unmap = false;
  DO_MAP_IF_NEEDED(srcFrame, &do_unmap);

  cvai_image_t tmp_image;
  memset(&tmp_image, 0, sizeof(cvai_image_t));
  cvai_face_info_t tmp_face_info;
  memset(&tmp_face_info, 0, sizeof(cvai_face_info_t));
  uint32_t h, w, s;
  uint8_t *p;
  if (CVIAI_SUCCESS != PREPARE_FACE_ALIGNMENT_DATA(srcFrame, &tmp_image, face_info, &tmp_face_info,
                                                   &h, &w, &s, &p)) {
    return CVIAI_FAILURE;
  }

  CVI_AI_CreateImage(dst_image, FACE_IMAGE_H, FACE_IMAGE_W, PIXEL_FORMAT_RGB_888);

  cv::Mat image(h, w, CV_8UC3, p, s);
  cv::Mat warp_image(cv::Size(dst_image->width, dst_image->height), image.type(), dst_image->pix[0],
                     dst_image->stride[0]);

  if (face_align(image, warp_image, tmp_face_info) != 0) {
    LOGE("face align failed.\n");
    return CVIAI_FAILURE;
  }

  free(tmp_face_info.pts.x);
  free(tmp_face_info.pts.y);

  DO_UNMAP_IF_NEEDED(srcFrame, do_unmap);

  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    CVI_AI_Free(&tmp_image);
  }

#if 0 /* for debug */
  cv::cvtColor(warp_image, warp_image, cv::COLOR_RGB2BGR);
  cv::imwrite("visual/aligned_face.jpg", warp_image);
#endif

  return CVIAI_SUCCESS;
}

CVI_S32 ALIGN_FACE_TO_FRAME(VIDEO_FRAME_INFO_S *srcFrame, VIDEO_FRAME_INFO_S *dstFrame,
                            cvai_face_info_t &face_info) {
  if (false == IS_SUPPORTED_FORMAT(srcFrame)) {
    return CVI_FAILURE;
  }
  bool do_unmap = false;
  DO_MAP_IF_NEEDED(srcFrame, &do_unmap);

  cvai_image_t tmp_image;
  memset(&tmp_image, 0, sizeof(cvai_image_t));
  cvai_face_info_t tmp_face_info;
  memset(&tmp_face_info, 0, sizeof(cvai_face_info_t));
  uint32_t h, w, s;
  uint8_t *p;
  if (CVI_SUCCESS != PREPARE_FACE_ALIGNMENT_DATA(srcFrame, &tmp_image, &face_info, &tmp_face_info,
                                                 &h, &w, &s, &p)) {
    return CVI_FAILURE;
  }

  cv::Mat image(h, w, CV_8UC3, p, s);
  cv::Mat warp_image(cv::Size(dstFrame->stVFrame.u32Width, dstFrame->stVFrame.u32Height),
                     image.type(), dstFrame->stVFrame.pu8VirAddr[0],
                     dstFrame->stVFrame.u32Stride[0]);

  if (face_align(image, warp_image, tmp_face_info) != 0) {
    LOGE("face_align failed.\n");
    return CVI_FAILURE;
  }

  free(tmp_face_info.pts.x);
  free(tmp_face_info.pts.y);

  CVI_AI_Free(&tmp_image);
  DO_UNMAP_IF_NEEDED(srcFrame, do_unmap);

  return CVI_SUCCESS;
}

}  // namespace cviai