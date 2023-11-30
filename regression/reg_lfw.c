#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

typedef int (*Inference)(const cvitdl_handle_t, VIDEO_FRAME_INFO_S *, cvtdl_face_t *);

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "Usage: reg_lfw <retina path> <model path> <pair_txt_path> <result file path> <is "
        "attribute model>.\n");
    printf("Pair txt format: lable image1_path image2_path.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_TDL_PerfettoInit();
  CVI_S32 ret = CVI_TDL_SUCCESS;

  CVI_TDL_SUPPORTED_MODEL_E model_id = atoi(argv[5]) == 1 ? CVI_TDL_SUPPORTED_MODEL_FACEATTRIBUTE
                                                          : CVI_TDL_SUPPORTED_MODEL_FACERECOGNITION;
  Inference inference = atoi(argv[5]) == 1 ? CVI_TDL_FaceAttribute : CVI_TDL_FaceRecognition;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&facelib_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_TDL_OpenModel(facelib_handle, model_id, argv[2]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_TDL_SetModelThreshold(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, 0.8);
  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_TDL_SetVpssTimeout(facelib_handle, -1);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum;
  ret = CVI_TDL_Eval_LfwInit(eval_handle, argv[3], true, &imageNum);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Lfw evaluation init failed with pairs file: %s!\n", argv[3]);
    return ret;
  }

  for (uint32_t i = 0; i < imageNum; i++) {
    char *name1 = NULL;
    char *name2 = NULL;
    int label;
    CVI_TDL_Eval_LfwGetImageLabelPair(eval_handle, i, &name1, &name2, &label);
    printf("[%d/%d] label %d: image1 %s image2 %s\n", i + 1, imageNum, label, name1, name2);

    VIDEO_FRAME_INFO_S frame1;
    CVI_S32 ret = CVI_TDL_ReadImage(name1, &frame1, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_face_t face1, face2;
    memset(&face1, 0, sizeof(cvtdl_face_t));
    memset(&face2, 0, sizeof(cvtdl_face_t));

    CVI_TDL_RetinaFace(facelib_handle, &frame1, &face1);
    if (face1.size > 0) {
      inference(facelib_handle, &frame1, &face1);
    } else {
      printf("cannot find face: %s\n", name1);
    }
    CVI_TDL_ReleaseImage(&frame1);

    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_TDL_ReadImage(name2, &frame2, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    CVI_TDL_RetinaFace(facelib_handle, &frame2, &face2);
    if (face2.size > 0) {
      inference(facelib_handle, &frame2, &face2);
    } else {
      printf("cannot find face: %s\n", name2);
    }

    if (face1.size > 0 && face2.size > 0) {
      CVI_TDL_Eval_LfwInsertFace(eval_handle, i, label, &face1, &face2);
    }

    CVI_TDL_Free(&face1);
    CVI_TDL_Free(&face2);
    CVI_TDL_ReleaseImage(&frame2);
    free(name1);
    free(name2);
  }

  CVI_TDL_Eval_LfwSave2File(eval_handle, argv[4]);
  CVI_TDL_Eval_LfwClearInput(eval_handle);
  CVI_TDL_Eval_LfwClearEvalData(eval_handle);

  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
