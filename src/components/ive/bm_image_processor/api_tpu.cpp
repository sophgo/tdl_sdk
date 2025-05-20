#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <assert.h>
#include <math.h>

#include "cvi_tpu.hpp"

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

typedef struct sg_cv_blend_2way{
	unsigned long long left_img_addr[3];
	int left_width[3];
	int left_height[3];
	unsigned long long right_img_addr[3];
	int right_width[3];
	int right_height[3];
	unsigned long long wgt_mem_addr[2];
	int overlay_lx;
	int overlay_rx;
	unsigned long long blend_img_addr[3];
	int blend_width[3];
	int blend_height[3];
	int channel;
	int format;
	int wgt_mode;
} __attribute__((packed)) sg_cv_blend_2way_t;

int subads_ref(unsigned char *input1, unsigned char *input2,
               unsigned char *output, int img_size)
{
	for (int i = 0; i < img_size; i++)
		output[i] = abs(input1[i] - input2[i]);

	return 0;
}

int threshold_ref(unsigned char *input, unsigned char *output, int height,
                  int width, TPU_THRESHOLD_TYPE threshold_type,
                  unsigned char threshold, unsigned char max_value)
{
	switch (threshold_type)
	{
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

int cpu_2way_blend(int lwidth, int lheight, unsigned char *left_img,
                   int rwidth, int rheight, unsigned char *right_img,
                   int bwidth, int bheight, unsigned char *blend_img,
                   int overlay_lx, int overlay_rx, unsigned char *wgt,
                   int channel, int format, int wgt_mode)
{
	unsigned short alpha = 0;

	int overlay_w = overlay_rx - overlay_lx + 1;
	int overlay_uv_w = overlay_w / 2;

	if (format == PIXEL_FORMAT_YUV_PLANAR_420) {
		// Y channel
		for (int y = 0; y < bheight; y++) {
			for (int x = 0; x < bwidth; x++) {
				if (x < overlay_lx) {
					blend_img[y * bwidth + x] = left_img[y * lwidth + x];
				} else if (x > overlay_rx) {
					int right_x = x - overlay_lx;
					if (right_x >= rwidth) right_x = rwidth - 1;

					blend_img[y * bwidth + x] = right_img[y * rwidth + right_x];
				} else {
					alpha = wgt[y * overlay_w + (x - overlay_lx)];
					unsigned char left_p = left_img[y * lwidth + x];

					int right_x = x - overlay_lx;
					if (right_x < 0) right_x = 0;
					if (right_x >= rwidth) right_x = rwidth - 1;
					unsigned char right_p = right_img[y * rwidth + right_x];

					blend_img[y * bwidth + x] = (alpha * left_p + (255 - alpha) * right_p) >> 8;
				}
			}
		}

		int uv_bheight = BM_ALIGN(bheight / 2, 2);
		int uv_bwidth = BM_ALIGN(bwidth / 2, 2);
		int buv_size = uv_bheight * uv_bwidth;

		int uv_lheight = BM_ALIGN(lheight / 2, 2);
		int uv_lwidth = BM_ALIGN(lwidth / 2, 2);
		int luv_size = uv_lheight * uv_lwidth;

		int uv_rheight = BM_ALIGN(rheight / 2, 2);
		int uv_rwidth = BM_ALIGN(rwidth / 2, 2);
		int ruv_size = uv_rheight * uv_rwidth;

		unsigned char *wgt_uv = NULL;
		if (wgt_mode == WGT_YUV_SHARE) {
			wgt_uv = (unsigned char*)malloc(uv_bheight * overlay_uv_w);
			for (int y = 0; y < uv_bheight; y++) {
				for (int x = 0; x < overlay_uv_w; x++) {
					int offset00 = overlay_w * y * 2 + x * 2;
					int offset01 = overlay_w * y * 2 + x * 2 + 1;
					int offset10 = overlay_w * (y * 2 + 1) + x * 2;
					int offset11 = overlay_w * (y * 2 + 1) + x * 2 + 1;
					wgt_uv[y * overlay_uv_w + x] =
							(wgt[offset00] + wgt[offset01] + wgt[offset10] + wgt[offset11]) >> 2;
				}
			}
		}

		// U V channel
		for (int y = 0; y < uv_bheight; y++) {
			for (int x = 0; x < uv_bwidth; x++) {
				int blend_u_offet = bwidth * bheight + y * uv_bwidth + x;
				int blend_v_offet = bwidth * bheight + buv_size + y * uv_bwidth + x;

				if (x < (overlay_lx/ 2)) {
					blend_img[blend_u_offet] = left_img[lwidth * lheight + y * uv_lwidth + x];
					blend_img[blend_v_offet] = left_img[lwidth * lheight + luv_size + y * uv_lwidth + x];
				} else if (x > (overlay_rx / 2)) {
					int right_x = x - overlay_lx/ 2;
					if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

					blend_img[blend_u_offet] = right_img[rwidth * rheight + y * uv_rwidth + right_x];
					blend_img[blend_v_offet] = right_img[rwidth * rheight + ruv_size + y * uv_rwidth + right_x];
				} else {
					// yuv alpha share
					if (wgt_mode == WGT_YUV_SHARE) {
						alpha = wgt_uv[y * overlay_uv_w + x - overlay_lx / 2];
					} else {
						alpha = wgt[bheight * overlay_w + y * overlay_uv_w + x - overlay_lx / 2];
					}

					unsigned char left_u = left_img[lwidth * lheight + y * uv_lwidth + x];
					unsigned char left_v = left_img[lwidth * lheight + luv_size + y * uv_lwidth + x];

					int right_x = x - overlay_lx/ 2;
					if (right_x < 0) right_x = 0;
					if (right_x >= uv_rwidth) right_x = uv_rwidth - 1;

					unsigned char right_u = right_img[rwidth * rheight + y * uv_rwidth + right_x];
					unsigned char right_v = right_img[rwidth * rheight + ruv_size + y * uv_rwidth + right_x];

					blend_img[blend_u_offet] = (alpha * left_u + (255 - alpha) * right_u) >> 8;
					blend_img[blend_v_offet] = (alpha * left_v + (255 - alpha) * right_v) >> 8;
				}
			}
		}

		free(wgt_uv);
	} else {
		for (int c = 0; c < channel; c++) {
			for (int y = 0; y < bheight; y++) {
				for (int x = 0; x < bwidth; x++) {
					if (x < overlay_lx) {
						blend_img[c * bheight * bwidth + y * bwidth + x] =
									left_img[c * lheight * lwidth + y * lwidth + x];
					} else if (x > overlay_rx) {
						int right_x = x - overlay_lx;
						if (right_x >= rwidth) right_x = rwidth - 1;
						blend_img[c * bheight * bwidth + y * bwidth + x] =
									right_img[c * rheight * rwidth + y * rwidth + right_x];
					} else {
						alpha = wgt[y * overlay_w + (x - overlay_lx)];
						unsigned char left_pixel = left_img[c * lheight * lwidth + y * lwidth + x];
						unsigned char right_pixel = right_img[c * rheight * rwidth + y * rwidth + (x - overlay_lx)];

						blend_img[c * bheight * bwidth + y * bwidth + x] =
								(alpha * left_pixel + (255 - alpha) * right_pixel) >> 8;
					}
				}
			}
		}
	}

	return 0;
}


bm_status_t sg_tpu_kernel_launch(bm_handle_t handle,
                                 const char  *func_name,
                                 void        *param,
                                 size_t      size,
                                 tpu_kernel_module_t tpu_module)
{
	// tpu_kernel_module_t tpu_module = NULL;
	tpu_kernel_function_t func_id = 0;
// #ifdef USE_CV184X
// 	tpu_module = tpu_kernel_load_module_file(handle, "/mnt/tpu_files/lib/libtpu_kernel_module.so");
// #else
// 	tpu_module = tpu_kernel_load_module_file(handle, "");
// #endif

	// if (!tpu_module) {
	// 	printf("%s:[ERROR] tpu kernel load module file failed\n", __func__);
	// 	return BM_ERR_FAILURE;
	// }

	func_id = tpu_kernel_get_function(handle, tpu_module, (char*)func_name);
	bm_status_t ret = tpu_kernel_launch(handle, func_id, param, size);

	// if (tpu_kernel_free_module(handle, tpu_module)) {
	// 	printf("%s:[ERROR] tpu module unload failed\n", __func__);
	// 	return BM_ERR_FAILURE;
	// }

	return ret;
}

bm_status_t tpu_subads_param_check(bm_handle_t handle, int height, int width, PIXEL_FORMAT_E format)
{
	if (handle == NULL) {
		bmlib_log("SUBADS", BMLIB_LOG_ERROR, "Can not get handle!\r\n");
		return BM_ERR_PARAM;
	}

	if (height < 1 || height > 4096) {
		bmlib_log("SUBADS", BMLIB_LOG_ERROR,
			"Invalid height(%d), the img height should between 2 and 4096!\r\n", height);
		return BM_ERR_PARAM;
	}

	if (width < 1 || width > 4096) {
		bmlib_log("SUBADS", BMLIB_LOG_ERROR,
			"Invalid width(%d), the img width should between 2 and 4096!\r\n", width);
		return BM_ERR_PARAM;
	}

	if (format != PIXEL_FORMAT_RGB_888_PLANAR &&
		format != PIXEL_FORMAT_BGR_888_PLANAR &&
		format != PIXEL_FORMAT_YUV_PLANAR_444 &&
		format != PIXEL_FORMAT_YUV_PLANAR_420 &&
		format != PIXEL_FORMAT_YUV_400) {
		bmlib_log("SUBADS", BMLIB_LOG_ERROR, "The img format(%d) not supported!\r\n", format);
		return BM_ERR_PARAM;
	}

	return BM_SUCCESS;
}

bm_status_t tpu_cv_subads(bm_handle_t handle,
                          CVI_S32 height,
                          CVI_S32 width,
                          CVI_S32 format,
                          CVI_S32 channel,
                          bm_device_mem_t *src1_mem,
                          bm_device_mem_t *src2_mem,
                          bm_device_mem_t *dst_mem,
                          tpu_kernel_module_t tpu_module)
{
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
			api.height[i] = BM_ALIGN(height / 2, 2);
			api.width[i] = BM_ALIGN(width / 2, 2);
		}
	} else {
		for (int c = 0; c < channel; c++) {
			api.height[c] = height;
			api.width[c] = width;
		}
	}

	for (int i = 0; i < channel; i++) {
		api.input1_addr[i] = bm_mem_get_device_addr(src1_mem[i]);
		api.input2_addr[i] = bm_mem_get_device_addr(src2_mem[i]);
		api.output_addr[i] = bm_mem_get_device_addr(dst_mem[i]);

		api.input1_str[i] = api.input2_str[i] = api.output_str[i] = api.width[i];
	}

	ret = sg_tpu_kernel_launch(handle, "cv_subads", &api, sizeof(api), tpu_module);
	if (ret != BM_SUCCESS) {
		bmlib_log("SUBADS", BMLIB_LOG_ERROR, "sg_tpu_kernel_launch!\r\n");
		return BM_ERR_FAILURE;
	}

	return ret;
}

bm_status_t tpu_threshold_check(bm_handle_t handle, CVI_S32 height,
                                CVI_S32 width, TPU_THRESHOLD_TYPE mode,
                                CVI_U32 threshold, CVI_U32 max_value)
{
	if (handle == NULL) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "Can not get handle!\r\n");
		return BM_ERR_PARAM;
	}

	if (height < 2 || height > 4096) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
				"Invalid height(%d), the img height should between 2 and 4096!\r\n", height);
		return BM_ERR_PARAM;
	}

	if (width < 2 || width > 4096) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
			"Invalid height(%d), the img height should between 2 and 4096!\r\n", height);
		return BM_ERR_PARAM;
	}

	if (mode != THRESHOLD_BINARY &&
		mode != THRESHOLD_BINARY_INV &&
		mode != THRESHOLD_TRUNC &&
		mode != THRESHOLD_TOZERO &&
		mode != THRESHOLD_TOZERO_INV) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
			"Invalid threshold type(%d), the threshold type should between 0 and 4!\r\n", mode);
		return BM_ERR_PARAM;
	}

	if (max_value > 255 || threshold > 255) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR,
			"Invalid threshold value(%d), the threshold value should between 0 and 255!\r\n", max_value);
		return BM_ERR_PARAM;
	}

	return BM_SUCCESS;
}

bm_status_t tpu_cv_threshold(bm_handle_t handle,
                             CVI_S32 height,
                             CVI_S32 width,
                             TPU_THRESHOLD_TYPE mode,
                             CVI_U32 threshold,
                             CVI_U32 max_value,
                             bm_device_mem_t *input_mem,
                             bm_device_mem_t *output_mem,
                             tpu_kernel_module_t tpu_module)
{
	bm_status_t ret = BM_ERR_FAILURE;

	ret = tpu_threshold_check(handle, height, width, mode, threshold, max_value);
	if (ret) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "Invalid Parameter!\r\n");
		return BM_ERR_PARAM;
	}

	sg_api_cv_threshold_t api;
	memset(&api, 0, sizeof(api));

	api.input_addr[0] = bm_mem_get_device_addr(*input_mem);
	api.output_addr[0] = bm_mem_get_device_addr(*output_mem);
	api.width[0] = api.input_str[0] = api.output_str[0] = width;
	api.height[0] = height;
	api.type = mode;
	api.thresh = threshold;
	api.max_value = max_value;
	api.channel = 1; // Only Support PIXEL_FORMAT_YUV_400

	ret = sg_tpu_kernel_launch(handle, "cv_threshold", &api, sizeof(api), tpu_module);
	if (ret != BM_SUCCESS) {
		bmlib_log("THRESHOLD", BMLIB_LOG_ERROR, "sg_tpu_kernel_launch!\r\n");
		return BM_ERR_FAILURE;
	}

	return ret;
}

bm_status_t tpu_blend_param_check(CVI_S32 lwidth, CVI_S32 lheight,
                                  CVI_S32 rwidth, CVI_S32 rheight,
                                  CVI_S32 bheight, CVI_S32 bwidth,
                                  CVI_S32 overlay_lx, CVI_S32 overlay_rx,
                                  PIXEL_FORMAT_E format, TPU_BLEND_WGT_MODE mode,
                                  bm_handle_t handle)
{
	if (handle == NULL) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "Can not get handel!\r\n");
		return BM_ERR_PARAM;
	}

	if (format != PIXEL_FORMAT_RGB_888_PLANAR &&
		format != PIXEL_FORMAT_BGR_888_PLANAR &&
		format != PIXEL_FORMAT_YUV_PLANAR_420 &&
		format != PIXEL_FORMAT_YUV_400) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The img format(%d) not supported!\r\n", format);
		return BM_ERR_PARAM;
	}

	if (lheight != rheight && lheight != bheight && rheight != bheight) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR,
			"The heights of the left image, right image, and blend image should be the same.\r\n");
		return BM_ERR_PARAM;
	}

	if (lheight < 8 || lheight > 4096 ||
		rheight < 8 || rheight > 4096 ||
		bheight < 8 || bheight > 4096) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The img height should be [8, 4096(%d)]\r\n", lwidth);
		return BM_ERR_PARAM;
	}

	if ((format == PIXEL_FORMAT_YUV_PLANAR_420) && (lwidth % 4 != 0)) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR,
			"In PIXEL_FORMAT_YUV_PLANAR_420 mode, the left image width should be 4-aligned\r\n");
		return BM_ERR_PARAM;
	}

	if ((format == PIXEL_FORMAT_YUV_PLANAR_420) && (rwidth % 4 != 0)) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR,
			"In PIXEL_FORMAT_YUV_PLANAR_420 mode, the right image width should be 4-aligned\r\n");
		return BM_ERR_PARAM;
	}

	if ((format == PIXEL_FORMAT_YUV_PLANAR_420) && (bwidth % 4 != 0)) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR,
			"In PIXEL_FORMAT_YUV_PLANAR_420 mode, the blend img width should be 4-aligned\r\n");
		return BM_ERR_PARAM;
	}

	if ((format == PIXEL_FORMAT_YUV_PLANAR_420) && (lheight % 2 != 0)) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR,
			"In PIXEL_FORMAT_YUV_PLANAR_420 mode, the img height should be 2-aligned\r\n");
		return BM_ERR_PARAM;
	}

	if (lwidth < 8 || rwidth < 8 || bwidth < 8) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The img height should be [0, lwidth(%d)]\r\n", lwidth);
	}

	if (overlay_lx < 0 || overlay_lx >= lwidth) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The overlay of left must be [0, lwidth(%d)]\r\n", lwidth);
		return BM_ERR_PARAM;
	}

	if (overlay_rx < 0 || overlay_rx >= lwidth) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The overlay of right must be [0, rwidth(%d)]\r\n", rwidth);
		return BM_ERR_PARAM;
	}

	if ((overlay_rx - overlay_lx + 1) < 0 || (overlay_rx - overlay_lx + 1) > 2000) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "The overlap width must be strictly between 0 and 2000 \r\n");
		return BM_ERR_PARAM;
	}

	if (format == PIXEL_FORMAT_YUV_PLANAR_420 && (overlay_rx - overlay_lx + 1) % 2 != 0) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "Img format(%d), the overlay width shoud be 2-aligned\r\n", format);
		return BM_ERR_PARAM;
	}

	if (mode != WGT_YUV_SHARE && mode != WGT_UV_SHARE) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "Invalid BLEND WGT MODE (%d)!\r\n", mode);
		return BM_ERR_PARAM;
	}

	return BM_SUCCESS;

}

bm_status_t tpu_2way_blending(bm_handle_t handle,
                              CVI_S32 lwidth,
                              CVI_S32 lheight,
                              CVI_S32 rwidth,
                              CVI_S32 rheight,
                              bm_device_mem_t *left_mem,
                              bm_device_mem_t *right_mem,
                              CVI_S32 blend_w,
                              CVI_S32 blend_h,
                              bm_device_mem_t *blend_mem,
                              CVI_S32 overlay_lx,
                              CVI_S32 overlay_rx,
                              bm_device_mem_t *wgt_phy_mem,
                              TPU_BLEND_WGT_MODE mode,
                              int format, int channel,
                              tpu_kernel_module_t tpu_module)
{
	bm_status_t ret;
	sg_cv_blend_2way_t api;
	memset(&api, 0, sizeof(sg_cv_blend_2way_t));

	ret = tpu_blend_param_check(lwidth, lheight, rwidth, rheight,
			blend_w, blend_h, overlay_lx, overlay_rx, (PIXEL_FORMAT_E)format, mode, handle);

	if (ret != BM_SUCCESS) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "Invalid Parameter!\r\n");
		return BM_ERR_PARAM;
	}

	for (int i = 0; i < channel; i++) {
		api.left_img_addr[i] = bm_mem_get_device_addr(left_mem[i]);
		api.right_img_addr[i] = bm_mem_get_device_addr(right_mem[i]);
		api.blend_img_addr[i] = bm_mem_get_device_addr(blend_mem[i]);
	}

	if ((overlay_rx - overlay_lx + 1) != 0) {
		api.wgt_mem_addr[0] = bm_mem_get_device_addr(wgt_phy_mem[0]); // YUV SHARE
		if (mode == WGT_UV_SHARE)
			api.wgt_mem_addr[1] = bm_mem_get_device_addr(wgt_phy_mem[1]); // UV SHARE
	}

	// api.wgt_mem_addr = bm_mem_get_device_addr(wgt_phy_mem[0]); // YUV SHARE
	api.overlay_lx = overlay_lx;
	api.overlay_rx = overlay_rx;
	api.channel = channel;
	api.format = format;
	api.wgt_mode = (int)mode;

	if (format == PIXEL_FORMAT_YUV_PLANAR_420) {
		api.left_height[0] = lheight;
		api.left_width[0] = lwidth;
		api.right_height[0] = rheight;
		api.right_width[0] = rwidth;
		api.blend_height[0] = blend_h;
		api.blend_width[0] = blend_w;

		for (int i = 1; i < 3; i++) {
			api.left_height[i] = BM_ALIGN(lheight / 2, 2);
			api.left_width[i] = BM_ALIGN(lwidth / 2, 2);
			api.right_height[i] = BM_ALIGN(rheight / 2, 2);
			api.right_width[i] = BM_ALIGN(rwidth / 2, 2);
			api.blend_height[i] = BM_ALIGN(blend_h / 2, 2);
			api.blend_width[i] = BM_ALIGN(blend_w / 2, 2);
		}
	} else {
		for (int i = 0; i < channel; i++) {
			api.left_height[i] = lheight;
			api.left_width[i] = lwidth;
			api.right_height[i] = rheight;
			api.right_width[i] = rwidth;
			api.blend_height[i] = blend_h;
			api.blend_width[i] = blend_w;
		}
	}

	ret = sg_tpu_kernel_launch(handle, "cv_blend_2way", &api, sizeof(api), tpu_module);
	if (ret != BM_SUCCESS) {
		bmlib_log("BLEND", BMLIB_LOG_ERROR, "sg_tpu_kernel_launch!\r\n");
		return BM_ERR_FAILURE;
	}

	return ret;
}
