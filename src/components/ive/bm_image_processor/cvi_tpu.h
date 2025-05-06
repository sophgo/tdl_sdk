#ifndef __CVI_TPU_H__
#define __CVI_TPU_H__

#include "cvi_comm_video.h"
#include "cvi_common.h"
#include "cvi_type.h"
#include "bmlib_runtime.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#define BM_ALIGN(x, a) (((x) + (a) - 1) / (a) * (a))

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
                          CVI_S32 format, CVI_S32 channel,
                          bm_device_mem_t *src1_mem, bm_device_mem_t *src2_mem,
                          bm_device_mem_t *dst_mem);

bm_status_t tpu_cv_threshold(bm_handle_t handle, CVI_S32 height, CVI_S32 width,
                             TPU_THRESHOLD_TYPE mode, CVI_U32 threshold,
                             CVI_U32 max_value, bm_device_mem_t *input_mem,
                             bm_device_mem_t *output_mem);

bm_status_t tpu_2way_blending(bm_handle_t handle, CVI_S32 lwidth,
                              CVI_S32 lheight, CVI_S32 rwidth, CVI_S32 rheight,
                              bm_device_mem_t *left_mem,
                              bm_device_mem_t *right_mem, CVI_S32 blend_w,
                              CVI_S32 blend_h, bm_device_mem_t *blend_mem,
                              CVI_S32 overlay_lx, CVI_S32 overlay_rx,
                              bm_device_mem_t *wgt_phy_mem,
                              TPU_BLEND_WGT_MODE mode, int format, int channel);

int cpu_2way_blend(int lwidth, int lheight, unsigned char *left_img, int rwidth,
                   int rheight, unsigned char *right_img, int bwidth,
                   int bheight, unsigned char *blend_img, int overlay_lx,
                   int overlay_rx, unsigned char *wgt, int channel, int format,
                   int wgt_mode);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif  // __CVI_TPU_H__
