#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int genFeatureFile(const char *img_dir, int *num, int *total) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(img_dir);

  int fail_num = 0;
  int idx = 0;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8) continue;
    char line[500] = "\0";
    strcat(line, img_dir);
    strcat(line, "/");
    strcat(line, entry->d_name);

    printf("%s\n", line);
    VIDEO_FRAME_INFO_S frFrame;
    int ret = CVI_TDL_ReadImage(img_handle, line, &frFrame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_face_t face;
    memset(&face, 0, sizeof(cvtdl_face_t));
    CVI_TDL_RetinaFace(facelib_handle, &frFrame, &face);
    if (face.size > 0) {
      CVI_TDL_Service_FaceAngleForAll(&face);
      CVI_TDL_FaceQuality(facelib_handle, &frFrame, &face, NULL);
    }

    if (face.size == 0 || face.info[0].face_quality < 0.5 ||
        fabs(face.info[0].head_pose.pitch) > 0.45 || fabs(face.info[0].head_pose.yaw) > 0.45) {
      fail_num++;
    }

    CVI_TDL_Free(&face);
    CVI_TDL_ReleaseImage(img_handle, &frFrame);
    printf("num: %d\n", idx);
    idx++;
  }

  *num = fail_num;
  *total = idx;
  closedir(dirp);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf(
        "Usage: %s <face detect model path> <face quality model path> \
           <pos_image_root_dir> <neg_image_root_dir>.\n",
        argv[0]);
    printf("Face detect model path: Path to face detect cvimodel.\n");
    printf("Face attribute model path: Path to face attribute cvimodel.\n");
    printf("Pos image root dir: Path to the positive data directory.\n");
    printf("Neg image root dir: Path to the negative data directory.\n");
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
  ret |= CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_FACEQUALITY, argv[2]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_SetSkipVpssPreprocess(facelib_handle, CVI_TDL_SUPPORTED_MODEL_RETINAFACE, false);

  int pos_num = 0, pos_total = 0;
  int neg_num = 0, neg_total = 0;
  genFeatureFile(argv[3], &pos_num, &pos_total);
  genFeatureFile(argv[4], &neg_num, &neg_total);

  printf("pos num: %d / pos total %d\n", pos_num, pos_total);
  printf("neg num: %d / neg total %d\n", neg_num, neg_total);

  CVI_TDL_DestroyHandle(facelib_handle);
}
