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

cvitdl_handle_t handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "Usage: %s <face detect model path> <liveness model path> \
           <root_dir> <pair_txt_path> <result_path>.\n",
        argv[0]);
    printf("Face detect model path: Path to face detect cvimodel.\n");
    printf("Liveness model path: Path to liveness cvimodel.\n");
    printf("Root dir: Image root directory.\n");
    printf("Pair txt path: Image list txt file path. <format: image1_path image2_path label>.\n");
    printf("Result path: Path to result file.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret = CVI_TDL_OpenModel(handle, CVI_TDL_SUPPORTED_MODEL_LIVENESS, argv[2]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, false);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum;
  CVI_TDL_Eval_LfwInit(eval_handle, argv[4], false, &imageNum);

  int idx = 0;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  for (uint32_t i = 0; i < imageNum; i++) {
    char *name1 = NULL;
    char *name2 = NULL;
    int label;
    CVI_TDL_Eval_LfwGetImageLabelPair(eval_handle, i, &name1, &name2, &label);

    char name1_full[500] = "\0";
    char name2_full[500] = "\0";

    strcat(name1_full, argv[3]);
    strcat(name1_full, "/");
    strcat(name1_full, name1);
    strcat(name2_full, argv[3]);
    strcat(name2_full, "/");
    strcat(name2_full, name2);
    free(name1);
    free(name2);

    VIDEO_FRAME_INFO_S frame1;
    // printf("name1_full: %s\n", name1_full);

    CVI_S32 ret = CVI_TDL_ReadImage(img_handle, name1_full, &frame1, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_TDL_ReadImage(img_handle, name2_full, &frame2, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_face_t rgb_face;
    memset(&rgb_face, 0, sizeof(cvtdl_face_t));

    cvtdl_face_t ir_face;
    memset(&ir_face, 0, sizeof(cvtdl_face_t));

    CVI_TDL_RetinaFace(handle, &frame1, &rgb_face);
    CVI_TDL_RetinaFace(handle, &frame2, &ir_face);

    if (rgb_face.size > 0) {
      if (ir_face.size > 0) {
        CVI_TDL_Liveness(handle, &frame1, &frame2, &rgb_face, &ir_face);
      } else {
        rgb_face.info[0].liveness_score = -2.0;
      }
      printf("label: %d, score: %f\n", label, rgb_face.info[0].liveness_score);
      CVI_TDL_Eval_LfwInsertLabelScore(eval_handle, idx, label, rgb_face.info[0].liveness_score);
      idx++;
    }

    CVI_TDL_Free(&rgb_face);
    CVI_TDL_Free(&ir_face);
    CVI_TDL_ReleaseImage(img_handle, &frame1);
    CVI_TDL_ReleaseImage(img_handle, &frame2);
  }

  CVI_TDL_Eval_LfwSave2File(eval_handle, argv[5]);
  CVI_TDL_Eval_LfwClearInput(eval_handle);
  CVI_TDL_Eval_LfwClearEvalData(eval_handle);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(handle);
  CVI_SYS_Exit();
}
