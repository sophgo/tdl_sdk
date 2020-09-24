#include <stdio.h>
#include <stdlib.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int prepareFeature(cviai_eval_handle_t eval_handle, bool is_query) {
  uint32_t num = 0;
  CVI_AI_Eval_Market1501GetImageNum(eval_handle, is_query, &num);
  for (int i = 0; i < num; ++i) {
    char *image = NULL;
    int cam_id;
    int pid;

    CVI_AI_Eval_Market1501GetPathIdPair(eval_handle, i, is_query, &image, &cam_id, &pid);

    VB_BLK blk;
    VIDEO_FRAME_INFO_S rgb_frame;
    CVI_S32 ret = CVI_AI_ReadImage(image, &blk, &rgb_frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_object_t obj;
    memset(&obj, 0, sizeof(cvai_object_t));
    obj.size = 1;
    obj.info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * obj.size);
    obj.width = rgb_frame.stVFrame.u32Width;
    obj.height = rgb_frame.stVFrame.u32Height;
    obj.info[0].bbox.x1 = 0;
    obj.info[0].bbox.y1 = 0;
    obj.info[0].bbox.x2 = rgb_frame.stVFrame.u32Width;
    obj.info[0].bbox.y2 = rgb_frame.stVFrame.u32Height;
    obj.info[0].bbox.score = 0.99;
    obj.info[0].classes = 0;
    memset(&obj.info[0].feature, 0, sizeof(cvai_feature_t));

    CVI_AI_OSNet(facelib_handle, &rgb_frame, &obj);
    CVI_AI_Eval_Market1501InsertFeature(eval_handle, i, is_query, &obj.info[0].feature);

    printf("image %s, cam %d, pid %d\n", image, cam_id, pid);
    free(image);

    CVI_VB_ReleaseBlock(blk);
  }

  return CVI_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <reid model path> <image_root_dir> <feature_dir>.\n", argv[0]);
    printf("Reid model path: Path to the reid cvimodel.\n");
    printf("Image root dir: Root directory to the test images.\n");
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_OSNET, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_OSNET, false);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_Eval_Market1501Init(eval_handle, argv[2]);

  prepareFeature(eval_handle, true);
  prepareFeature(eval_handle, false);

  CVI_AI_Eval_Market1501EvalCMC(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
