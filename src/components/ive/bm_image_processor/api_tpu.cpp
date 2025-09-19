#include <assert.h>
#include <cvi_math.h>
#include <limits.h>
#include <math.h>
#include "cvi_tpu.hpp"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "utils/tdl_log.hpp"

typedef struct sg_api_cv_subads {
  int channel;
  unsigned long long input1_addr[3];
  unsigned long long input2_addr[3];
  unsigned long long output_addr[3];
  int width[3];
  int height[3];
  int input1_str[3];
  int input2_str[3];
  int output_str[3];
} __attribute__((packed)) sg_api_cv_subads_t;

typedef struct {
  int channel;
  unsigned long long input_addr[3];
  unsigned long long output_addr[3];
  int width[3];
  int height[3];
  int input_str[3];
  int output_str[3];
  int type;
  unsigned int thresh;
  unsigned int max_value;
} __attribute__((packed)) sg_api_cv_threshold_t;

typedef struct sg_cv_blend_2way {
  unsigned long long left_img_addr[3];
  int left_width[3];
  int left_stride[3];
  int left_height[3];
  unsigned long long right_img_addr[3];
  int right_width[3];
  int right_stride[3];
  int right_height[3];
  unsigned long long wgt_mem_addr[2];
  int overlay_lx;
  int overlay_rx;
  unsigned long long blend_img_addr[3];
  int blend_width[3];
  int blend_stride[3];
  int blend_height[3];
  int channel;
  int format;
  int wgt_mode;
} __attribute__((packed)) sg_cv_blend_2way_t;

typedef struct sg_api_cv_morph {
  unsigned long long src_addr;
  unsigned long long dst_addr;
  int width;
  int height;
  int kh;
  int kw;
  int stride_i;
  int stride_o;
  int op;
} __attribute__((packed)) sg_api_cv_morph_t;

static unsigned char blend_pixel(unsigned char left, unsigned char right,
                                 unsigned char alpha) {
  float r = (alpha * left + (255 - alpha) * right) / 255;

  return (r > 255) ? 255 : (unsigned char)r;
}

static void process_uv_plane(unsigned char *blend_uv_base,
                             unsigned char *left_uv_base,
                             unsigned char *right_uv_base, int format,
                             int bheight, int bwidth, int uv_bstride,
                             int lheight, int uv_lstride, int rheight,
                             int rwidth, int uv_rstride, unsigned char *wgt,
                             TPU_BLEND_WGT_MODE wgt_mode, int overlay_lx,
                             int overlay_rx) {
  unsigned char *wgt_uv = NULL;
  unsigned char alpha_u = 0, alpha_v = 0;
  int overlay_w = overlay_rx - overlay_lx + 1;
  int overlay_uv_w = overlay_w / 2;

  int uv_bheight = ALIGN(bheight / 2, 2);
  int uv_bwidth = ALIGN(bwidth / 2, 2);

  int uv_lheight = ALIGN(lheight / 2, 2);

  int uv_rheight = ALIGN(rheight / 2, 2);
  int uv_rwidth = ALIGN(rwidth / 2, 2);

  int buv_size = uv_bheight * uv_bstride;
  int luv_size = uv_lheight * uv_lstride;
  int ruv_size = uv_rheight * uv_rstride;

  if (wgt_mode == WGT_YUV_SHARE) {
    wgt_uv = (unsigned char *)malloc(uv_bheight * overlay_uv_w);
    for (int y = 0; y < uv_bheight; y++) {
      for (int x = 0; x < overlay_uv_w; x++) {
        int offset00 = overlay_w * y * 2 + x * 2;
        int offset01 = overlay_w * y * 2 + x * 2 + 1;
        int offset10 = overlay_w * (y * 2 + 1) + x * 2;
        int offset11 = overlay_w * (y * 2 + 1) + x * 2 + 1;
        wgt_uv[y * overlay_uv_w + x] =
            (wgt[offset00] + wgt[offset01] + wgt[offset10] + wgt[offset11]) >>
            2;
      }
    }
  }

  for (int y = 0; y < uv_bheight; y++) {
    for (int x = 0; x < uv_bwidth; x++) {
      int blend_u_offet = y * uv_bstride + x;
      int blend_v_offet = buv_size + y * uv_bstride + x;

      if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
        blend_u_offet =
            y * uv_bstride + (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
        blend_v_offet =
            y * uv_bstride + (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
      }

      if (x < (overlay_lx / 2)) {
        int left_u_pos = y * uv_lstride + x;
        int left_v_pos = luv_size + y * uv_lstride + x;

        if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
          left_u_pos = y * uv_lstride +
                       (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
          left_v_pos = y * uv_lstride +
                       (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
        }

        blend_uv_base[blend_u_offet] = left_uv_base[left_u_pos];
        blend_uv_base[blend_v_offet] = left_uv_base[left_v_pos];

      } else if (x > (overlay_rx / 2)) {
        int right_x = x - overlay_lx / 2;
        if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

        int right_u_pos = y * uv_rstride + right_x;
        int right_v_pos = ruv_size + y * uv_rstride + right_x;

        if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
          right_u_pos =
              y * uv_rstride +
              (format == PIXEL_FORMAT_NV12 ? right_x * 2 : right_x * 2 + 1);
          right_v_pos =
              y * uv_rstride +
              (format == PIXEL_FORMAT_NV12 ? right_x * 2 + 1 : right_x * 2);
        }

        blend_uv_base[blend_u_offet] = right_uv_base[right_u_pos];
        blend_uv_base[blend_v_offet] = right_uv_base[right_v_pos];
      } else {
        if (wgt_mode == WGT_YUV_SHARE) {
          alpha_u = alpha_v = wgt_uv[y * overlay_uv_w + (x - overlay_lx / 2)];
        } else if (wgt_mode == WGT_UV_SHARE) {
          if (format == PIXEL_FORMAT_YUV_PLANAR_420) {
            alpha_u = alpha_v = wgt[bheight * overlay_w + y * overlay_uv_w +
                                    (x - overlay_lx / 2)];
          } else {
            int alpha_ux = format == PIXEL_FORMAT_NV12
                               ? 2 * (x - overlay_lx / 2)
                               : 2 * (x - overlay_lx / 2) + 1;
            int alpha_vx = format == PIXEL_FORMAT_NV12
                               ? 2 * (x - overlay_lx / 2) + 1
                               : 2 * (x - overlay_lx / 2);

            alpha_u = wgt[overlay_w * bheight + y * overlay_w + alpha_ux];
            alpha_v = wgt[overlay_w * bheight + y * overlay_w + alpha_vx];
          }
        }

        int left_u_pos = y * uv_lstride + x;
        int left_v_pos = luv_size + y * uv_lstride + x;

        if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
          left_u_pos = y * uv_lstride +
                       (format == PIXEL_FORMAT_NV12 ? 2 * x : 2 * x + 1);
          left_v_pos = y * uv_lstride +
                       (format == PIXEL_FORMAT_NV12 ? 2 * x + 1 : 2 * x);
        }

        unsigned char left_u = left_uv_base[left_u_pos];
        unsigned char left_v = left_uv_base[left_v_pos];

        int right_x = x - overlay_lx / 2;
        if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

        int right_u_pos = y * uv_rstride + right_x;
        int right_v_pos = ruv_size + y * uv_rstride + right_x;

        if (format == PIXEL_FORMAT_NV12 || format == PIXEL_FORMAT_NV21) {
          right_u_pos =
              y * uv_rstride +
              (format == PIXEL_FORMAT_NV12 ? right_x * 2 : right_x * 2 + 1);
          right_v_pos =
              y * uv_rstride +
              (format == PIXEL_FORMAT_NV12 ? right_x * 2 + 1 : right_x * 2);
        }

        unsigned char right_u = right_uv_base[right_u_pos];
        unsigned char right_v = right_uv_base[right_v_pos];

        blend_uv_base[blend_u_offet] = blend_pixel(left_u, right_u, alpha_u);
        blend_uv_base[blend_v_offet] = blend_pixel(left_v, right_v, alpha_v);
      }
    }
  }

  if (wgt_mode == WGT_YUV_SHARE) free(wgt_uv);
}

/*cmodel*/

int subads_ref(unsigned char *input1, unsigned char *input2,
               unsigned char *output, int img_size) {
  for (int i = 0; i < img_size; i++) output[i] = abs(input1[i] - input2[i]);

  return 0;
}

int threshold_ref(unsigned char *input, unsigned char *output, int height,
                  int width, TPU_THRESHOLD_TYPE threshold_type,
                  unsigned char threshold, unsigned char max_value) {
  switch (threshold_type) {
    case THRESHOLD_BINARY:
      for (int i = 0; i < width * height; i++) {
        if (input[i] > threshold)
          output[i] = max_value;
        else
          output[i] = 0;
      }
      break;
    case THRESHOLD_BINARY_INV:
      for (int i = 0; i < width * height; i++) {
        if (input[i] > threshold)
          output[i] = 0;
        else
          output[i] = max_value;
      }
      break;
    case THRESHOLD_TRUNC:
      for (int i = 0; i < width * height; i++) {
        if (input[i] > threshold)
          output[i] = threshold;
        else
          output[i] = input[i];
      }
      break;
    case THRESHOLD_TOZERO:
      for (int i = 0; i < width * height; i++) {
        if (input[i] > threshold)
          output[i] = input[i];
        else
          output[i] = 0;
      }
      break;
    case THRESHOLD_TOZERO_INV:
      for (int i = 0; i < width * height; i++) {
        if (input[i] > threshold)
          output[i] = 0;
        else
          output[i] = input[i];
      }
      break;
    default:
      break;
  }

  return 0;
}

int cpu_2way_blend(int lwidth, int lheight, int *left_stride,
                   unsigned char *left_img, int rwidth, int rheight,
                   int *right_stride, unsigned char *right_img, int bwidth,
                   int bheight, int *blend_stride, unsigned char *blend_img,
                   int overlay_lx, int overlay_rx, unsigned char *wgt,
                   TPU_BLEND_WGT_MODE wgt_mode, int channel, int format) {
  unsigned char alpha = 0;

  if (lheight != rheight) {
    printf("left_img right img height");
  }
  printf("lwidth %d\n", lwidth);

  int overlay_w = overlay_rx - overlay_lx + 1;
  int lstride, luv_stride, rstride, ruv_stride, bstride, buv_stride;

  if (left_stride != NULL) {
    lstride = left_stride[0];
    if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
        format == PIXEL_FORMAT_NV21)
      luv_stride = left_stride[1];
  }

  if (right_stride != NULL) {
    rstride = right_stride[0];
    if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
        format == PIXEL_FORMAT_NV21)
      ruv_stride = right_stride[1];
  }

  if (blend_stride != NULL) {
    bstride = blend_stride[0];
    if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
        format == PIXEL_FORMAT_NV21)
      buv_stride = blend_stride[1];
  }

  if (format == PIXEL_FORMAT_YUV_PLANAR_420 || format == PIXEL_FORMAT_NV12 ||
      format == PIXEL_FORMAT_NV21) {
    // Y channel
    for (int y = 0; y < bheight; y++) {
      for (int x = 0; x < bwidth; x++) {
        if (x < overlay_lx) {
          blend_img[y * bstride + x] = left_img[y * lstride + x];
        } else if (x > overlay_rx) {
          int right_x = x - overlay_lx;
          if (right_x >= rwidth) right_x = rwidth - 1;

          blend_img[y * bstride + x] = right_img[y * rstride + right_x];
        } else {
          alpha = wgt[y * overlay_w + (x - overlay_lx)];
          unsigned char left_p = left_img[y * lstride + x];

          int right_x = x - overlay_lx;
          if (right_x < 0) right_x = 0;
          if (right_x >= rwidth) right_x = rwidth - 1;
          unsigned char right_p = right_img[y * rstride + right_x];

          blend_img[y * bstride + x] = blend_pixel(left_p, right_p, alpha);
        }
      }
    }

    process_uv_plane(blend_img + bheight * bstride,
                     left_img + lheight * lstride,
                     right_img + rheight * rstride, format, bheight, bwidth,
                     buv_stride, lheight, luv_stride, rheight, rwidth,
                     ruv_stride, wgt, wgt_mode, overlay_lx, overlay_rx);
  } else {
    for (int c = 0; c < channel; c++) {
      for (int y = 0; y < bheight; y++) {
        for (int x = 0; x < bwidth; x++) {
          if (x < overlay_lx) {
            blend_img[c * bheight * bstride + y * bstride + x] =
                left_img[c * lheight * lstride + y * lstride + x];
          } else if (x > overlay_rx) {
            int right_x = x - overlay_lx;
            if (right_x >= rwidth) right_x = rwidth - 1;
            blend_img[c * bheight * bstride + y * bstride + x] =
                right_img[c * rheight * rstride + y * rstride + right_x];
          } else {
            alpha = wgt[y * overlay_w + (x - overlay_lx)];
            unsigned char left_pixel =
                left_img[c * lheight * lstride + y * lstride + x];
            unsigned char right_pixel =
                right_img[c * rheight * rstride + y * rstride +
                          (x - overlay_lx)];

            blend_img[c * bheight * bstride + y * bstride + x] =
                blend_pixel(left_pixel, right_pixel, alpha);
          }
        }
      }
    }
  }

  return 0;
}

bm_status_t tpu_subads_param_check(bm_handle_t handle, int height, int width,
                                   PIXEL_FORMAT_E format) {
  if (handle == NULL) {
    bmlib_log("SUBADS", BMLIB_LOG_ERROR, "Can not get handle!\r\n");
    return BM_ERR_PARAM;
  }

  if (height < 1 || height > 4096) {
    bmlib_log(
        "SUBADS", BMLIB_LOG_ERROR,
        "Invalid height(%d), the img height should between 2 and 4096!\r\n",
        height);
    return BM_ERR_PARAM;
  }

  if (width < 1 || width > 4096) {
    bmlib_log("SUBADS", BMLIB_LOG_ERROR,
              "Invalid width(%d), the img width should between 2 and 4096!\r\n",
              width);
    return BM_ERR_PARAM;
  }

  if (format != PIXEL_FORMAT_RGB_888_PLANAR &&
      format != PIXEL_FORMAT_BGR_888_PLANAR &&
      format != PIXEL_FORMAT_YUV_PLANAR_444 &&
      format != PIXEL_FORMAT_YUV_PLANAR_420 && format != PIXEL_FORMAT_YUV_400) {
    bmlib_log("SUBADS", BMLIB_LOG_ERROR,
              "The img format(%d) not supported!\r\n", format);
    return BM_ERR_PARAM;
  }

  return BM_SUCCESS;
}

bm_status_t tpu_cv_subads(bm_handle_t handle, CVI_S32 height, CVI_S32 width,
                          PIXEL_FORMAT_E format, CVI_S32 channel,
                          bm_device_mem_t *src1_mem, bm_device_mem_t *src2_mem,
                          bm_device_mem_t *dst_mem,
                          tpu_kernel_module_t tpu_module,
                          tpu_kernel_function_t func_id) {
  bm_status_t ret = BM_ERR_FAILURE;

  ret = tpu_subads_param_check(handle, height, width, (PIXEL_FORMAT_E)format);
  if (ret != BM_SUCCESS) {
    bmlib_log("SUBADS", BMLIB_LOG_ERROR, "Invalid Parameter!\r\n");
    return BM_ERR_PARAM;
  }

  sg_api_cv_subads_t api;
  memset(&api, 0, sizeof(sg_api_cv_subads_t));

  api.channel = channel;

  if (format == PIXEL_FORMAT_YUV_PLANAR_420) {
    api.height[0] = height;
    api.width[0] = width;

    for (int i = 1; i < 3; i++) {
      api.height[i] = ALIGN(height / 2, 2);
      api.width[i] = ALIGN(width / 2, 2);
    }
  } else {
    for (int c = 0; c < channel; c++) {
      api.height[c] = height;
      api.width[c] = width;
    }
  }

  for (int i = 0; i < channel; i++) {
    src1_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    src2_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    dst_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    api.input1_addr[i] = bm_mem_get_device_addr(src1_mem[i]);
    api.input2_addr[i] = bm_mem_get_device_addr(src2_mem[i]);
    api.output_addr[i] = bm_mem_get_device_addr(dst_mem[i]);

    api.input1_str[i] = api.input2_str[i] = api.output_str[i] = api.width[i];
  }

  ret = tpu_kernel_launch(handle, func_id, &api, sizeof(api));
  if (ret != BM_SUCCESS) {
    bmlib_log("SUBADS", BMLIB_LOG_ERROR, "tpu_kernel_launch!\r\n");
    return BM_ERR_FAILURE;
  }

  return ret;
}

bm_status_t tpu_threshold_check(bm_handle_t handle, CVI_S32 height,
                                CVI_S32 width, TPU_THRESHOLD_TYPE mode,
                                CVI_U32 threshold, CVI_U32 max_value) {
  if (handle == NULL) {
    bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "Can not get handle!\r\n");
    return BM_ERR_PARAM;
  }

  if (height < 2 || height > 4096) {
    bmlib_log(
        "THRESHOLD", BMLIB_LOG_ERROR,
        "Invalid height(%d), the img height should between 2 and 4096!\r\n",
        height);
    return BM_ERR_PARAM;
  }

  if (width < 2 || width > 4096) {
    bmlib_log(
        "THRESHOLD", BMLIB_LOG_ERROR,
        "Invalid height(%d), the img height should between 2 and 4096!\r\n",
        height);
    return BM_ERR_PARAM;
  }

  if (mode != THRESHOLD_BINARY && mode != THRESHOLD_BINARY_INV &&
      mode != THRESHOLD_TRUNC && mode != THRESHOLD_TOZERO &&
      mode != THRESHOLD_TOZERO_INV) {
    bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
              "Invalid threshold type(%d), the threshold type should between 0 "
              "and 4!\r\n",
              mode);
    return BM_ERR_PARAM;
  }

  if (max_value > 255 || threshold > 255) {
    bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
              "Invalid threshold value(%d), the threshold value should between "
              "0 and 255!\r\n",
              max_value);
    return BM_ERR_PARAM;
  }

  return BM_SUCCESS;
}

bm_status_t tpu_cv_threshold(bm_handle_t handle, CVI_S32 height, CVI_S32 width,
                             TPU_THRESHOLD_TYPE mode, CVI_U32 threshold,
                             CVI_U32 max_value, bm_device_mem_t *input_mem,
                             bm_device_mem_t *output_mem,
                             tpu_kernel_module_t tpu_module,
                             tpu_kernel_function_t func_id) {
  bm_status_t ret = BM_ERR_FAILURE;

  ret = tpu_threshold_check(handle, height, width, mode, threshold, max_value);
  if (ret) {
    bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "Invalid Parameter!\r\n");
    return BM_ERR_PARAM;
  }

  sg_api_cv_threshold_t api;
  memset(&api, 0, sizeof(api));
  input_mem->flags.u.mem_type = BM_MEM_TYPE_DEVICE;
  output_mem->flags.u.mem_type = BM_MEM_TYPE_DEVICE;
  api.input_addr[0] = bm_mem_get_device_addr(*input_mem);
  api.output_addr[0] = bm_mem_get_device_addr(*output_mem);
  api.width[0] = api.input_str[0] = api.output_str[0] = width;
  api.height[0] = height;
  api.type = mode;
  api.thresh = threshold;
  api.max_value = max_value;
  api.channel = 1;  // Only Support PIXEL_FORMAT_YUV_400

  ret = tpu_kernel_launch(handle, func_id, &api, sizeof(api));
  if (ret != BM_SUCCESS) {
    bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "tpu_kernel_launch!\r\n");
    return BM_ERR_FAILURE;
  }

  return ret;
}

static int get_channel_info(int size[3], int stride[3], int h[3], int w[3],
                            int *channel, int width, int height, int format,
                            int dsize) {
  int ret = 0;

  switch (format) {
    case PIXEL_FORMAT_RGB_888_PLANAR:
    case PIXEL_FORMAT_BGR_888_PLANAR:
    case PIXEL_FORMAT_YUV_PLANAR_444:
      for (int i = 0; i < 3; i++) {
        size[i] = width * height;
        stride[i] = width * dsize;
        w[i] = width;
        h[i] = height;
      }
      *channel = 3;
      break;
    case PIXEL_FORMAT_YUV_PLANAR_420:
      size[0] = width * height;
      stride[0] = width * dsize;
      w[0] = width;
      h[0] = height;

      for (int i = 1; i < 3; i++) {
        size[i] = ALIGN(width / 2, 2) * ALIGN(height / 2, 2);
        stride[i] = ALIGN(width / 2, 2) * dsize;
        w[i] = ALIGN(width / 2, 2);
        h[i] = ALIGN(height / 2, 2);
      }

      *channel = 3;
      break;
    case PIXEL_FORMAT_NV12:
    case PIXEL_FORMAT_NV21:
      size[0] = width * height;
      size[1] = ALIGN(height / 2, 2) * width;

      stride[0] = stride[1] = width * dsize;
      stride[2] = 0;

      w[1] = w[0] = width;

      h[0] = height;
      h[1] = ALIGN(height / 2, 2);

      *channel = 2;
      break;
    case PIXEL_FORMAT_YUV_400:
      stride[0] = width * dsize;
      w[0] = width;
      h[0] = height;
      size[0] = width * height;

      *channel = 1;
      break;
    default:
      printf("%s: Img Format is not Supported\n", __func__);
      ret = -1;
      break;
  }

  return ret;
}

int set_blend_Image_param(ImageInfo *img, PIXEL_FORMAT_E img_format, int width,
                          std::vector<uint32_t> &w_stride, int height,
                          int dsize) {
  int channel_stride[3] = {0}, stride[3] = {0}, w[3] = {0}, h[3] = {0},
      channel = 3;
  int ret = 0;

  if (get_channel_info(channel_stride, stride, h, w, &channel, width, height,
                       img_format, dsize) != 0) {
    printf("%s: blend get_channel info failed\n", __func__);
    return -1;
  }

  img->format = img_format;
  img->channel = channel;

  if (!w_stride.empty()) {
    for (int i = 0; i < img->channel && i < w_stride.size(); i++) {
      if (w_stride[i] >= stride[i]) {
        stride[i] = w_stride[i];
      } else {
        assert(w_stride[i] > 0 && ((w_stride[i] & (w_stride[i] - 1)) == 0));
        int stride_temp = stride[i];
        stride[i] = ALIGN(stride_temp * dsize, w_stride[i]);
      }
      channel_stride[i] = stride[i] * h[i];
    }
  }

  for (int i = 0; i < channel; i++) {
    img->stride[i] = stride[i];
    img->channel_stride[i] = channel_stride[i];
    img->width[i] = w[i];
    img->height[i] = h[i];
  }

  return ret;
}

bm_status_t tpu_blend_param_check(bm_handle_t handle, ImageInfo *left_img,
                                  ImageInfo *right_img, ImageInfo *blend_img,
                                  short overlay_lx, short overlay_rx,
                                  TPU_BLEND_WGT_MODE mode) {
  if (handle == NULL) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "Can not get handel!\r\n");
    return BM_ERR_FAILURE;
  }

  if (left_img->format != right_img->format &&
      left_img->format != blend_img->format) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "Images formats must be same!\r\n");
    return BM_ERR_PARAM;
  }

  if (blend_img->format != PIXEL_FORMAT_RGB_888_PLANAR &&
      blend_img->format != PIXEL_FORMAT_YUV_PLANAR_420 &&
      blend_img->format != PIXEL_FORMAT_YUV_400 &&
      blend_img->format != PIXEL_FORMAT_NV12 &&
      blend_img->format != PIXEL_FORMAT_NV21) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "The img format(%d) not supported!\r\n",
              blend_img->format);
    return BM_ERR_PARAM;
  }

  if (left_img->height[0] != right_img->height[0] &&
      left_img->height[0] != blend_img->height[0]) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The heights of the left image, right image, and blend image "
              "should be the same.\r\n");
    return BM_ERR_PARAM;
  }

  if (left_img->width[0] < 8 || left_img->width[0] > 4096) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Left img width should be [8, 4096(%d)]\r\n",
              left_img->width[0]);
    return BM_ERR_PARAM;
  }

  if (left_img->height[0] < 8 || left_img->height[0] > 4096) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Left img height should be [8, 4096(%d)]\r\n",
              left_img->height[0]);
    return BM_ERR_PARAM;
  }

  if (right_img->width[0] < 8 || right_img->width[0] > 4096) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Right img width should be [8, 4096(%d)]\r\n",
              right_img->width[0]);
    return BM_ERR_PARAM;
  }

  if (right_img->height[0] < 8 || right_img->height[0] > 4096) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Right img height should be [8, 4096(%d)]\r\n",
              right_img->height[0]);
    return BM_ERR_PARAM;
  }

  if (blend_img->width[0] < 8) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Blend img width should be [8, 4096(%d)]\r\n",
              blend_img->width[0]);
    return BM_ERR_PARAM;
  }

  if (blend_img->height[0] < 8 || blend_img->height[0] > 4096) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The Blend img height should be [8, 4096(%d)]\r\n",
              blend_img->height[0]);
    return BM_ERR_PARAM;
  }

  bool is_target_format = (blend_img->format == PIXEL_FORMAT_YUV_PLANAR_420 ||
                           blend_img->format == PIXEL_FORMAT_NV12 ||
                           blend_img->format == PIXEL_FORMAT_NV21);

  if (is_target_format && (blend_img->height[0] % 2 != 0)) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "In PIXEL_FORMAT_YUV_PLANAR_420 mode, the img height should be "
              "2-aligned\r\n");
    return BM_ERR_PARAM;
  }

  if (is_target_format && (left_img->width[0] % 4 != 0)) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "In YUV420P or YUV420SP mode, the left image width should be "
              "4-aligned\r\n");
    return BM_ERR_PARAM;
  }

  if (is_target_format && (right_img->width[0] % 4 != 0)) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "In YUV420P or YUV420SP mode, the right image width should be "
              "4-aligned\r\n");
    return BM_ERR_PARAM;
  }

  if (is_target_format && (blend_img->width[0] % 4 != 0)) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "In YUV420P or YUV420SP mode, the blend image width should be "
              "4-aligned\r\n");
    return BM_ERR_PARAM;
  }

  if (overlay_lx < 0 || overlay_lx > left_img->width[0]) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The overlay of left must be [0, lwidth(%d)]\r\n",
              left_img->width[0]);
    return BM_ERR_PARAM;
  }

  if (overlay_rx < 0 || overlay_rx >= blend_img->width[0]) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The overlay of right must be [0, rwidth(%d)]\r\n",
              blend_img->width[0]);
    return BM_ERR_PARAM;
  }

  if ((overlay_rx - overlay_lx + 1) < 0 ||
      (overlay_rx - overlay_lx + 1) > 2000) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "The overlap width must be strictly between 0 and 2000 \r\n");
    return BM_ERR_PARAM;
  }

  if (is_target_format && ((overlay_rx - overlay_lx + 1) % 2 != 0)) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR,
              "Img format(%d), the overlay width shoud be 2-aligned\r\n",
              blend_img->format);
    return BM_ERR_PARAM;
  }

  if (mode != WGT_YUV_SHARE && mode != WGT_UV_SHARE) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "Invalid BLEND WGT MODE (%d)!\r\n",
              mode);
    return BM_ERR_PARAM;
  }

  return BM_SUCCESS;
}

bm_status_t tpu_2way_blending(bm_handle_t handle, ImageInfo *left_img,
                              bm_device_mem_t *left_mem, ImageInfo *right_img,
                              bm_device_mem_t *right_mem, ImageInfo *blend_img,
                              bm_device_mem_t *blend_mem, short overlay_lx,
                              short overlay_rx, bm_device_mem_t *wgt_phy_mem,
                              TPU_BLEND_WGT_MODE mode,
                              tpu_kernel_module_t tpu_module,
                              tpu_kernel_function_t func_id) {
  bm_status_t ret = BM_ERR_FAILURE;
  sg_cv_blend_2way_t api;
  memset(&api, 0, sizeof(sg_cv_blend_2way_t));
  ret = tpu_blend_param_check(handle, left_img, right_img, blend_img,
                              overlay_lx, overlay_rx, mode);
  if (ret != BM_SUCCESS) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "Invalid Parameter!\r\n");
    return BM_ERR_PARAM;
  }

  api.overlay_lx = overlay_lx;
  api.overlay_rx = overlay_rx;
  api.channel = blend_img->channel;
  api.format = blend_img->format;
  api.wgt_mode = (int)mode;

  if ((overlay_rx - overlay_lx + 1) != 0) {
    wgt_phy_mem[0].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    api.wgt_mem_addr[0] = bm_mem_get_device_addr(wgt_phy_mem[0]);
    if (mode == WGT_UV_SHARE) {
      wgt_phy_mem[1].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
      api.wgt_mem_addr[1] = bm_mem_get_device_addr(wgt_phy_mem[1]);
    }
  }
  for (int i = 0; i < blend_img->channel; i++) {
    api.left_height[i] = left_img->height[i];
    api.left_width[i] = left_img->width[i];
    api.left_stride[i] = left_img->stride[i];
    left_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    api.left_img_addr[i] = bm_mem_get_device_addr(left_mem[i]);

    api.right_height[i] = right_img->height[i];
    api.right_width[i] = right_img->width[i];
    api.right_stride[i] = right_img->stride[i];
    right_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    api.right_img_addr[i] = bm_mem_get_device_addr(right_mem[i]);

    api.blend_height[i] = blend_img->height[i];
    api.blend_width[i] = blend_img->width[i];
    api.blend_stride[i] = blend_img->stride[i];
    blend_mem[i].flags.u.mem_type = BM_MEM_TYPE_DEVICE;
    api.blend_img_addr[i] = bm_mem_get_device_addr(blend_mem[i]);
  }
  ret = tpu_kernel_launch(handle, func_id, &api, sizeof(api));
  if (ret != BM_SUCCESS) {
    bmlib_log("BLEND", BMLIB_LOG_ERROR, "tpu_kernel_launch!\r\n");
    return BM_ERR_FAILURE;
  }

  return ret;
}

void grayscale_dilate(const unsigned char *src, unsigned char *dst, int width,
                      int height, int w_stride, int ksize) {
  const int radius = ksize / 2;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char max_val = 0;

      for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int ny = y + ky;
          int nx = x + kx;

          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (src[ny * w_stride + nx] > max_val) {
              max_val = src[ny * w_stride + nx];
            }
          }
        }
      }
      dst[y * w_stride + x] = max_val;
    }
  }
}

void grayscale_erode(const unsigned char *src, unsigned char *dst, int width,
                     int height, int w_stride, int ksize) {
  const int radius = ksize / 2;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      unsigned char min_val = UCHAR_MAX;

      for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int ny = y + ky;
          int nx = x + kx;

          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (src[ny * w_stride + nx] < min_val) {
              min_val = src[ny * w_stride + nx];
            }
          }
        }
      }
      dst[y * w_stride + x] = min_val;
    }
  }
}

bm_status_t bm_cv_morph_check(bm_handle_t handle, PIXEL_FORMAT_E format,
                              int img_height, int img_width, int kw, int kh,
                              enum MorphTypes op) {
  const char *info_label =
      (op == MORPH_DILATE ? "MORPH_DILATE" : "MORPH_ERODE");
  if (handle == NULL) {
    bmlib_log(info_label, BMLIB_LOG_ERROR, "Can not get handle!\r\n");
    return BM_ERR_PARAM;
  }

  if (format != PIXEL_FORMAT_YUV_400) {
    bmlib_log(info_label, BMLIB_LOG_ERROR,
              "Only Support PIXEL_FORMAT_YUV_400!\r\n");
    return BM_ERR_PARAM;
  }

  if (op != MORPH_DILATE && op != MORPH_ERODE) {
    bmlib_log(info_label, BMLIB_LOG_ERROR, "MorphTypes(%d) Not Support!\r\n",
              op);
    return BM_ERR_PARAM;
  }

  if (kw > 7 || kh > 7) {
    bmlib_log(info_label, BMLIB_LOG_ERROR,
              "The kernel size must be not greater than 7!\r\n");
    return BM_ERR_PARAM;
  }

  if (img_height > 1440 && img_height < 5) {
    bmlib_log(info_label, BMLIB_LOG_ERROR,
              "the img height should between 8 and 1440!\r\n");
    return BM_ERR_PARAM;
  }

  if (img_width + kw - 1 > 2700 && img_width + kw - 1 < 5) {
    bmlib_log(info_label, BMLIB_LOG_ERROR, "image width is too large!\r\n");
    return BM_ERR_PARAM;
  }

  return BM_SUCCESS;
}

bm_status_t bm_cv_morph(bm_handle_t handle, bm_device_mem_t src_mem,
                        bm_device_mem_t dst_mem, int kh, int kw, int img_height,
                        int img_width, int img_w_stride, enum MorphTypes op,
                        tpu_kernel_module_t tpu_module,
                        tpu_kernel_function_t func_id) {
  bm_status_t ret = BM_SUCCESS;

  sg_api_cv_morph api;
  memset(&api, 0, sizeof(sg_api_cv_morph));
  src_mem.flags.u.mem_type = BM_MEM_TYPE_DEVICE;
  dst_mem.flags.u.mem_type = BM_MEM_TYPE_DEVICE;
  api.src_addr = bm_mem_get_device_addr(src_mem);
  api.dst_addr = bm_mem_get_device_addr(dst_mem);
  api.width = img_width;
  api.height = img_height;
  api.stride_i = img_w_stride;
  api.stride_o = img_w_stride;
  api.op = op;
  api.kh = kh;
  api.kw = kw;
  ret = tpu_kernel_launch(handle, func_id, &api, sizeof(api));

  if (ret != BM_SUCCESS) {
    bmlib_log("MORPH", BMLIB_LOG_ERROR, "tpu_kernel_launch!\r\n");
    return BM_ERR_FAILURE;
  }

  return ret;
}

bm_status_t bm_cv_dilate(bm_handle_t handle, bm_device_mem_t src_mem,
                         bm_device_mem_t dst_mem, PIXEL_FORMAT_E format,
                         int width, int height, int w_stride, int kw, int kh,
                         tpu_kernel_module_t tpu_module,
                         tpu_kernel_function_t func_id) {
  bm_status_t ret =
      bm_cv_morph_check(handle, format, height, width, kw, kh, MORPH_DILATE);
  if (ret != BM_SUCCESS) {
    return ret;
  }

  return bm_cv_morph(handle, src_mem, dst_mem, kh, kw, height, width, w_stride,
                     MORPH_DILATE, tpu_module, func_id);
}

bm_status_t bm_cv_erode(bm_handle_t handle, bm_device_mem_t src_mem,
                        bm_device_mem_t dst_mem, PIXEL_FORMAT_E format,
                        int width, int height, int w_stride, int kw, int kh,
                        tpu_kernel_module_t tpu_module,
                        tpu_kernel_function_t func_id) {
  bm_status_t ret =
      bm_cv_morph_check(handle, format, height, width, kw, kh, MORPH_ERODE);
  if (ret != BM_SUCCESS) {
    return ret;
  }

  return bm_cv_morph(handle, src_mem, dst_mem, kh, kw, height, width, w_stride,
                     MORPH_ERODE, tpu_module, func_id);
}
