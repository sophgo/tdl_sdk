#ifndef __CVI_TPU_H__
#define __CVI_TPU_H__

#include <cvi_comm_video.h>
#include <cvi_common.h>
#include <vector>
#include "bmlib_runtime.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

typedef enum _TPU_THRESHOLD_TYPE {
  THRESHOLD_BINARY = 0,
  THRESHOLD_BINARY_INV,
  THRESHOLD_TRUNC,
  THRESHOLD_TOZERO,
  THRESHOLD_TOZERO_INV
} TPU_THRESHOLD_TYPE;

typedef enum _TPU_BLEND_WGT_MODE {
  WGT_YUV_SHARE = 0,
  WGT_UV_SHARE
} TPU_BLEND_WGT_MODE;

bm_status_t tpu_cv_subads(bm_handle_t handle, CVI_S32 height, CVI_S32 width,
                          PIXEL_FORMAT_E format, CVI_S32 channel,
                          bm_device_mem_t *src1_mem, bm_device_mem_t *src2_mem,
                          bm_device_mem_t *dst_mem,
                          tpu_kernel_module_t tpu_module,
                          tpu_kernel_function_t func_id);

bm_status_t tpu_cv_threshold(bm_handle_t handle, CVI_S32 height, CVI_S32 width,
                             TPU_THRESHOLD_TYPE mode, CVI_U32 threshold,
                             CVI_U32 max_value, bm_device_mem_t *input_mem,
                             bm_device_mem_t *output_mem,
                             tpu_kernel_module_t tpu_module,
                             tpu_kernel_function_t func_id);

typedef struct Image_t {
  int channel;
  PIXEL_FORMAT_E format;
  int width[3];
  int height[3];
  int stride[3];
  int channel_stride[3];
} ImageInfo;

int set_blend_Image_param(ImageInfo *img, PIXEL_FORMAT_E img_format, int width,
                          std::vector<uint32_t> &w_stride, int height,
                          int dsize);

bm_status_t tpu_2way_blending(bm_handle_t handle, ImageInfo *left_img,
                              bm_device_mem_t *left_mem, ImageInfo *right_img,
                              bm_device_mem_t *right_mem, ImageInfo *blend_img,
                              bm_device_mem_t *blend_mem, short overlay_lx,
                              short overlay_rx, bm_device_mem_t *wgt_phy_mem,
                              TPU_BLEND_WGT_MODE mode,
                              tpu_kernel_module_t tpu_module,
                              tpu_kernel_function_t func_id);

enum MorphTypes {
  MORPH_ERODE = 0,   //!< see #erode
  MORPH_DILATE = 1,  //!< see #dilate
};

bm_status_t bm_cv_dilate(bm_handle_t handle, bm_device_mem_t src_mem,
                         bm_device_mem_t dst_mem, PIXEL_FORMAT_E format,
                         int width, int height, int w_stride, int kw, int kh,
                         tpu_kernel_module_t tpu_module,
                         tpu_kernel_function_t func_id);

bm_status_t bm_cv_erode(bm_handle_t handle, bm_device_mem_t src_mem,
                        bm_device_mem_t dst_mem, PIXEL_FORMAT_E format,
                        int width, int height, int w_stride, int kw, int kh,
                        tpu_kernel_module_t tpu_module,
                        tpu_kernel_function_t func_id);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif  // __CVI_TPU_H__
