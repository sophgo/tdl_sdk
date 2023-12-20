#include <stdio.h>
#include <stdlib.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

cvitdl_handle_t facelib_handle = NULL;

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <thermal model path> <image root folder> <evaluate json> <result json>.\n",
           argv[0]);
    return CVI_TDL_FAILURE;
  }

  CVI_TDL_PerfettoInit();
  CVI_S32 ret = CVI_TDL_SUCCESS;
  CVI_S32 vpssgrp_width = 1280;
  CVI_S32 vpssgrp_height = 720;

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

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model thermalface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, false);
  CVI_TDL_SetModelThreshold(facelib_handle, CVI_TDL_SUPPORTED_MODEL_THERMALFACE, 0.05);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_TDL_Eval_CocoInit(eval_handle, argv[2], argv[3], &image_num);
  CVI_TDL_Eval_CocoStartEval(eval_handle, argv[4]);
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_TDL_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);
    printf("Reading image %s\n", filename);
    VIDEO_FRAME_INFO_S frame;
    if (CVI_TDL_ReadImage(img_handle, filename, &frame, PIXEL_FORMAT_RGB_888) != CVI_TDL_SUCCESS) {
      printf("Read image [%s] failed.\n", filename);
      return CVI_TDL_FAILURE;
    }
    free(filename);

    cvtdl_face_t face;
    memset(&face, 0, sizeof(cvtdl_face_t));
    CVI_TDL_ThermalFace(facelib_handle, &frame, &face);

    cvtdl_object_t obj;
    obj.size = face.size;
    obj.info = (cvtdl_object_info_t *)malloc(sizeof(cvtdl_object_info_t) * obj.size);
    obj.width = -1;
    obj.height = -1;

    memset(obj.info, 0, sizeof(cvtdl_object_info_t) * obj.size);
    for (int i = 0; i < obj.size; i++) {
      obj.info[i].bbox = face.info[i].bbox;
      obj.info[i].classes = 0;
    }

    CVI_TDL_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_TDL_Free(&face);
    CVI_TDL_Free(&obj);
    CVI_TDL_ReleaseImage(img_handle, &frame);
  }
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_Eval_CocoEndEval(eval_handle);

  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
