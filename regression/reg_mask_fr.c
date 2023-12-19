#include <math.h>
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

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "Usage: reg_lfw <retina path> <mask fr path> <root_dir> <pair_txt_path> <result file "
        "path>.\n");
    printf("Retina path: Path to retina face cvimodel.\n");
    printf("Mask fr path: Path to mask face recognition cvimodel.\n");
    printf("Root dir: Path to image root directory.\n");
    printf("Pair txt path: Path to label/image pair list txt.\n");
    printf("Result file path: Path to result txt.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_TDL_PerfettoInit();
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

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_MASKFACERECOGNITION, argv[2]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, false);

  cvitdl_eval_handle_t eval_handle;
  ret = CVI_TDL_Eval_CreateHandle(&eval_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create Eval handle failed with %#x!\n", ret);
    return ret;
  }
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  uint32_t imageNum;
  CVI_TDL_Eval_LfwInit(eval_handle, argv[4], true, &imageNum);
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

    printf("label %d: image1 %s image2 %s\n", label, name1_full, name2_full);

    VIDEO_FRAME_INFO_S frame1;
    CVI_S32 ret = CVI_TDL_ReadImage(img_handle, name1_full, &frame1, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_TDL_ReadImage(img_handle, name2_full, &frame2, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_face_t face1, face2;
    memset(&face1, 0, sizeof(cvtdl_face_t));
    memset(&face2, 0, sizeof(cvtdl_face_t));

    CVI_TDL_RetinaFace(facelib_handle, &frame1, &face1);
    CVI_TDL_RetinaFace(facelib_handle, &frame2, &face2);

    if (face1.size > 0 && face2.size > 0) {
      CVI_TDL_MaskFaceRecognition(facelib_handle, &frame1, &face1);
      CVI_TDL_MaskFaceRecognition(facelib_handle, &frame2, &face2);

      CVI_TDL_Eval_LfwInsertFace(eval_handle, i, label, &face1, &face2);
    }

    CVI_TDL_Free(&face1);
    CVI_TDL_Free(&face2);
    CVI_TDL_ReleaseImage(img_handle, &frame1);
    CVI_TDL_ReleaseImage(img_handle, &frame2);
  }

  CVI_TDL_Eval_LfwSave2File(eval_handle, argv[5]);
  CVI_TDL_Eval_LfwClearInput(eval_handle);
  CVI_TDL_Eval_LfwClearEvalData(eval_handle);

  CVI_TDL_Eval_DestroyHandle(eval_handle);
  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
