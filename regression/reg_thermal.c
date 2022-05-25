#include <stdio.h>
#include <stdlib.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

cviai_handle_t facelib_handle = NULL;

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <thermal model path> <image root folder> <evaluate json> <result json>.\n",
           argv[0]);
    return CVIAI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;
  CVI_S32 vpssgrp_width = 1280;
  CVI_S32 vpssgrp_height = 720;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model thermalface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, false);
  CVI_AI_SetModelThreshold(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, 0.05);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_AI_Eval_CocoInit(eval_handle, argv[2], argv[3], &image_num);
  CVI_AI_Eval_CocoStartEval(eval_handle, argv[4]);
  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_AI_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);
    printf("Reading image %s\n", filename);
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &frame, PIXEL_FORMAT_RGB_888) != CVIAI_SUCCESS) {
      printf("Read image [%s] failed.\n", filename);
      return CVIAI_FAILURE;
    }
    free(filename);

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_ThermalFace(facelib_handle, &frame, &face);

    cvai_object_t obj;
    obj.size = face.size;
    obj.info = (cvai_object_info_t *)malloc(sizeof(cvai_object_info_t) * obj.size);
    obj.width = -1;
    obj.height = -1;

    memset(obj.info, 0, sizeof(cvai_object_info_t) * obj.size);
    for (int i = 0; i < obj.size; i++) {
      obj.info[i].bbox = face.info[i].bbox;
      obj.info[i].classes = 0;
    }

    CVI_AI_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_AI_Free(&face);
    CVI_AI_Free(&obj);
    CVI_AI_ReleaseImage(&frame);
  }

  CVI_AI_Eval_CocoEndEval(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
