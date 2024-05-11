#define _GNU_SOURCE
#include <dirent.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;
void dms_init(cvtdl_face_t *face) {
  cvtdl_dms_t *dms = (cvtdl_dms_t *)malloc(sizeof(cvtdl_dms_t));
  dms->reye_score = 0;
  dms->leye_score = 0;
  dms->yawn_score = 0;
  dms->phone_score = 0;
  dms->smoke_score = 0;
  dms->landmarks_106.size = 0;
  dms->landmarks_5.size = 0;
  dms->head_pose.yaw = 0;
  dms->head_pose.pitch = 0;
  dms->head_pose.roll = 0;
  dms->dms_od.info = NULL;
  dms->dms_od.size = 0;
  face->dms = dms;
}

static int run(const char *img_dir) {
  DIR *dirp;
  struct dirent *entry;
  dirp = opendir(img_dir);
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    char line[500] = "\0";
    strcat(line, img_dir);
    strcat(line, "/");
    strcat(line, entry->d_name);

    printf("%s\n", line);
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_TDL_SUCCESS;
    ret = CVI_TDL_ReadImage(img_handle, line, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvtdl_face_t face;
    memset(&face, 0, sizeof(cvtdl_face_t));
    dms_init(&face);

    CVI_TDL_IncarObjectDetection(facelib_handle, &frame, &face);

    CVI_TDL_Free(&face);
    CVI_TDL_ReleaseImage(img_handle, &frame);
  }
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  closedir(dirp);

  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <incarod_path> <img_dir>.\n", argv[0]);
    printf("Od path: Incar od model path.\n");
    printf("Img dir: Directory of dataset.\n");
    return CVI_TDL_FAILURE;
  }

  CVI_S32 ret = CVI_TDL_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_CreateHandle(&facelib_handle);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_INCAROBJECTDETECTION, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  run(argv[2]);

  CVI_TDL_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();

  return CVI_TDL_SUCCESS;
}
