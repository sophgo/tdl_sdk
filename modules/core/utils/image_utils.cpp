#include "image_utils.hpp"

#include "core/utils/vpss_helper.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "cviai_log.hpp"
#include "rescale_utils.hpp"

#define FACE_IMAGE_H 112
#define FACE_IMAGE_W 112
#define DEFAULT_ALIGN_WIDTH 6 /* NOTE: Make sure it align with DEFAULT_ALIGN, ex. 2^6 = 64 */

static uint32_t get_aligned_size(uint32_t width) {
  uint32_t width_align = ((width) >> DEFAULT_ALIGN_WIDTH) << DEFAULT_ALIGN_WIDTH;
  if (width_align < width) {
    return width_align + (1 >> DEFAULT_ALIGN_WIDTH);
  } else {
    return width_align;
  }
}

// TODO: move this function to cviai_types_mem
static void create_image(cvai_image_t *image, uint32_t h, uint32_t w, bool align_size) {
  /* NOTE: Support RGB PACKED */
  if (image->pix != NULL) {
    free(image->pix);
  }
  image->height = h;
  image->width = w;
  if (align_size) {
    image->stride = get_aligned_size(image->width * 3);  // TODO: wrong answer... check this
  } else {
    image->stride = image->width * 3;
  }
  image->pix = (uint8_t *)malloc(image->stride * image->height);
  memset(image->pix, 0, image->stride * image->height);
}

namespace cviai {

uint32_t get_image_size(cvai_image_t *dst) { return dst->stride * dst->height; }

uint32_t estimate_image_size(uint32_t h, uint32_t w, bool align_size) {
  uint32_t stride;
  if (align_size) {
    stride = get_aligned_size(w * 3);  // TODO: wrong answer... check this
  } else {
    stride = w * 3;
  }
  return stride * h;
}

int crop_image(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image, cvai_bbox_t *bbox) {
  printf("crop_image\n");
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVI_FAILURE;
  }
  bool do_unmap = false;
  size_t frame_size = srcFrame->stVFrame.u32Length[0] + srcFrame->stVFrame.u32Length[1] +
                      srcFrame->stVFrame.u32Length[2];
  if (srcFrame->stVFrame.pu8VirAddr[0] == NULL) {
    srcFrame->stVFrame.pu8VirAddr[0] =
        (CVI_U8 *)CVI_SYS_MmapCache(srcFrame->stVFrame.u64PhyAddr[0], frame_size);
    do_unmap = true;
  }
  uint32_t x1 = (uint32_t)roundf(bbox->x1);
  uint32_t y1 = (uint32_t)roundf(bbox->y1);
  uint32_t x2 = (uint32_t)roundf(bbox->x2);
  uint32_t y2 = (uint32_t)roundf(bbox->y2);
  uint32_t height = y2 - y1 + 1;
  uint32_t width = x2 - x1 + 1;
  create_image(dst_image, height, width, false);

  /* NOTE: Support RGB PACKED */
  CVI_U32 stride_frame = srcFrame->stVFrame.u32Stride[0];
  size_t copy_size = dst_image->stride * sizeof(uint8_t);
  CVI_U16 t = 0;
  for (CVI_U16 i = y1; i <= y2; i++) {
    memcpy(dst_image->pix + t * dst_image->stride,
           srcFrame->stVFrame.pu8VirAddr[0] + i * stride_frame + x1 * 3, copy_size);
    t += 1;
  }

  if (do_unmap) {
    CVI_SYS_Munmap((void *)srcFrame->stVFrame.pu8VirAddr[0], srcFrame->stVFrame.u32Length[0]);
    srcFrame->stVFrame.pu8VirAddr[0] = NULL;
    srcFrame->stVFrame.pu8VirAddr[1] = NULL;
    srcFrame->stVFrame.pu8VirAddr[2] = NULL;
  }

  return CVI_SUCCESS;
}

static VIDEO_FRAME_INFO_S g_wrap_frame;

int crop_image_face(VIDEO_FRAME_INFO_S *srcFrame, cvai_image_t *dst_image,
                    cvai_face_info_t *face_info, bool align) {
  if (srcFrame->stVFrame.enPixelFormat != PIXEL_FORMAT_RGB_888) {
    LOGE("Error: pixel format not match PIXEL_FORMAT_RGB_888.\n");
    return CVI_FAILURE;
  }
  if (!align) {
    return crop_image(srcFrame, dst_image, &face_info->bbox);
  }
  if (g_wrap_frame.stVFrame.u32Height == 0) {
    CREATE_ION_HELPER(&g_wrap_frame, FACE_IMAGE_W, FACE_IMAGE_H, PIXEL_FORMAT_RGB_888, "tpu");
  }
  create_image(dst_image, FACE_IMAGE_H, FACE_IMAGE_W, false);

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
    return CVI_FAILURE;
  }

  uint32_t stride_frame = g_wrap_frame.stVFrame.u32Stride[0];
  size_t copy_size = dst_image->stride * sizeof(uint8_t);
  for (uint32_t i = 0; i < FACE_IMAGE_H; i++) {
    memcpy(dst_image->pix + i * dst_image->stride,
           g_wrap_frame.stVFrame.pu8VirAddr[0] + i * stride_frame, copy_size);
  }

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

  return CVI_SUCCESS;
}

}  // namespace cviai