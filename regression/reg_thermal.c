#include <stdio.h>
#include <stdlib.h>

#include "cviai.h"
#include "core/utils/vpss_helper.h"

cviai_handle_t facelib_handle = NULL;

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <thermal model path> <image root folder> <evaluate json>.\n", argv[0]);
    return CVI_FAILURE;
  }

  CVI_S32 ret = CVI_SUCCESS;
  CVI_S32 vpssgrp_width = 1280;
  CVI_S32 vpssgrp_height = 720;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_BGR_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_BGR_888);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model thermalface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_THERMALFACE, false);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t image_num;
  CVI_AI_Eval_CocoInit(eval_handle, argv[2], argv[3], &image_num);

  for (uint32_t i = 0; i < image_num; i++) {
    char *filename = NULL;
    int id = 0;
    CVI_AI_Eval_CocoGetImageIdPair(eval_handle, i, &filename, &id);
    printf("Reading image %s\n", filename);
    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    if (CVI_AI_ReadImage(filename, &blk, &frame, PIXEL_FORMAT_RGB_888) != CVI_SUCCESS) {
      printf("Read image failed.\n");
      break;
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

    for (int i = 0; i < obj.size; i++) {
      obj.info[i].bbox = face.info[i].bbox;
      obj.info[i].classes = 0;
    }

    CVI_AI_Eval_CocoInsertObject(eval_handle, id, &obj);
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk);
  }

  CVI_AI_Eval_CocoSave2Json(eval_handle, "result.json");
  CVI_AI_Eval_CocoClearInput(eval_handle);
  CVI_AI_Eval_CocoClearObject(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
