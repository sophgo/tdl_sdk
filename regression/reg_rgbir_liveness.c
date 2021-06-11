#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cvimath/cvimath.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

cviai_handle_t handle = NULL;

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
    return CVI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 3, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 3);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret = CVI_AI_SetModelPath(handle, CVI_AI_SUPPORTED_MODEL_LIVENESS, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  cviai_eval_handle_t eval_handle;
  ret = CVI_AI_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }

  uint32_t imageNum;
  CVI_AI_Eval_LfwInit(eval_handle, argv[4], false, &imageNum);

  int idx = 0;
  for (uint32_t i = 0; i < imageNum; i++) {
    char *name1 = NULL;
    char *name2 = NULL;
    int label;
    CVI_AI_Eval_LfwGetImageLabelPair(eval_handle, i, &name1, &name2, &label);

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

    VB_BLK blk1;
    VIDEO_FRAME_INFO_S frame1;
    // printf("name1_full: %s\n", name1_full);
    CVI_S32 ret = CVI_AI_ReadImage(name1_full, &blk1, &frame1, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    VB_BLK blk2;
    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_AI_ReadImage(name2_full, &blk2, &frame2, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t rgb_face;
    memset(&rgb_face, 0, sizeof(cvai_face_t));

    cvai_face_t ir_face;
    memset(&ir_face, 0, sizeof(cvai_face_t));

    CVI_AI_RetinaFace(handle, &frame1, &rgb_face);
    CVI_AI_RetinaFace(handle, &frame2, &ir_face);

    if (rgb_face.size > 0) {
      if (ir_face.size > 0) {
        CVI_AI_Liveness(handle, &frame1, &frame2, &rgb_face, &ir_face);
      } else {
        rgb_face.info[0].liveness_score = -2.0;
      }
      printf("label: %d, score: %f\n", label, rgb_face.info[0].liveness_score);
      CVI_AI_Eval_LfwInsertLabelScore(eval_handle, idx, label, rgb_face.info[0].liveness_score);
      idx++;
    }

    CVI_AI_Free(&rgb_face);
    CVI_AI_Free(&ir_face);
    CVI_VB_ReleaseBlock(blk1);
    CVI_VB_ReleaseBlock(blk2);
  }

  CVI_AI_Eval_LfwSave2File(eval_handle, argv[5]);
  CVI_AI_Eval_LfwClearInput(eval_handle);
  CVI_AI_Eval_LfwClearEvalData(eval_handle);

  CVI_AI_Eval_DestroyHandle(eval_handle);
  CVI_AI_DestroyHandle(handle);
  CVI_SYS_Exit();
}
