#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/utils/vpss_helper.h"
#include "cvi_kit.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

cvitdl_handle_t tdl_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <Topformer model path> <input image path> <downsampling ratio>.\n", argv[0]);
    printf("Topformer model path: Path to Topformer cvimodel.\n");
    printf("Input image path: Path to input images for segmentation.\n");
    printf("Downsampling ratio: Downsampling ratio of result images.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  // DOWN_RATO
  int down_rato = atoi(argv[3]);
  ret = CVI_TDL_Set_Segmentation_DownRato(tdl_handle, CVI_TDL_SUPPORTED_MODEL_TOPFORMER_SEG,
                                          down_rato);
  if (ret != CVI_SUCCESS) {
    printf("Set topformer downrato file %#x!\n", ret);
    return ret;
  }
  // MODEL_DIR
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_TOPFORMER_SEG, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model topformer failed with %#x!\n", ret);
    return ret;
  }

  char* input_image_path = argv[2];
  imgprocess_t img_handle;
  int outH, outW;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  VIDEO_FRAME_INFO_S rgb_frame;
  cvtdl_seg_t seg_ann;

  ret = CVI_TDL_ReadImage(img_handle, input_image_path, &rgb_frame, PIXEL_FORMAT_RGB_888_PLANAR);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  }

  outH = (int)ceil((float)rgb_frame.stVFrame.u32Height / down_rato);
  outW = (int)ceil((float)rgb_frame.stVFrame.u32Width / down_rato);

  ret = CVI_TDL_Topformer_Seg(tdl_handle, &rgb_frame, &seg_ann);
  if (ret != CVI_SUCCESS) {
    printf("Segmentation failed with %#x!\n", ret);
  } else {
    for (int x = 0; x < outH; ++x) {
      for (int y = 0; y < outW; ++y) {
        printf("%d ", (int)seg_ann.class_id[x * outW + y]);
      }
      printf("\n");
    }
  }
  CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  CVI_TDL_FreeSeg(&seg_ann);

  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_DestroyHandle(tdl_handle);
  return CVI_SUCCESS;
}
