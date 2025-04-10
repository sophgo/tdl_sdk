#include <stdio.h>
#include <stdlib.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int prepareFeature(cvitdl_eval_handle_t eval_handle, bool is_query) {
  uint32_t num = 0;
  CVI_TDL_Eval_Market1501GetImageNum(eval_handle, is_query, &num);
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  for (int i = 0; i < num; ++i) {
    char *image = NULL;
    int cam_id;
    int pid;

    CVI_TDL_Eval_Market1501GetPathIdPair(eval_handle, i, is_query, &image, &cam_id, &pid);

    VIDEO_FRAME_INFO_S rgb_frame;
    CVI_S32 ret = CVI_TDL_ReadImage(img_handle, image, &rgb_frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_object_t obj;
    memset(&obj, 0, sizeof(cvtdl_object_t));
    obj.size = 1;
    obj.info = (cvtdl_object_info_t *)malloc(sizeof(cvtdl_object_info_t) * obj.size);
    memset(obj.info, 0, sizeof(cvtdl_object_info_t) * obj.size);
    obj.width = rgb_frame.stVFrame.u32Width;
    obj.height = rgb_frame.stVFrame.u32Height;
    obj.info[0].bbox.x1 = 0;
    obj.info[0].bbox.y1 = 0;
    obj.info[0].bbox.x2 = rgb_frame.stVFrame.u32Width;
    obj.info[0].bbox.y2 = rgb_frame.stVFrame.u32Height;
    obj.info[0].bbox.score = 0.99;
    obj.info[0].classes = 0;
    memset(&obj.info[0].feature, 0, sizeof(cvtdl_feature_t));

    CVI_TDL_OSNet(facelib_handle, &rgb_frame, &obj);
    CVI_TDL_Eval_Market1501InsertFeature(eval_handle, i, is_query, &obj.info[0].feature);

    printf("image %s, cam %d, pid %d\n", image, cam_id, pid);
    free(image);

    CVI_TDL_Free(&obj);
    CVI_TDL_ReleaseImage(img_handle, &rgb_frame);
  }
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <reid model path> <image_root_dir>.\n", argv[0]);
    printf("Reid model path: Path to the reid cvimodel.\n");
    printf("Image root dir: Root directory to the test images.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&facelib_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_OSNET, false);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_Eval_Market1501Init(eval_handle, argv[2]);

  prepareFeature(eval_handle, true);
  prepareFeature(eval_handle, false);

  CVI_TDL_Eval_Market1501EvalCMC(eval_handle);

  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
