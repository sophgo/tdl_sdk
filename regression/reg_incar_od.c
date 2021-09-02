#define _GNU_SOURCE
#include <dirent.h>
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "cviai_perfetto.h"
#include "evaluation/cviai_evaluation.h"
#include "evaluation/cviai_media.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;
void dms_init(cvai_face_t *face) {
  cvai_dms_t *dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
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

  while ((entry = readdir(dirp)) != NULL) {
    if (entry->d_type != 8 && entry->d_type != 0) continue;
    char line[500] = "\0";
    strcat(line, img_dir);
    strcat(line, "/");
    strcat(line, entry->d_name);

    printf("%s\n", line);
    VB_BLK blk;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVIAI_SUCCESS;
    ret = CVI_AI_ReadImage(line, &blk, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVIAI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    dms_init(&face);

    CVI_AI_IncarObjectDetection(facelib_handle, &frame, &face);

    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk);
  }
  closedir(dirp);

  return CVIAI_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <incarod_path> <img_dir>.\n", argv[0]);
    printf("Od path: Incar od model path.\n");
    printf("Img dir: Directory of dataset.\n");
    return CVIAI_FAILURE;
  }

  CVI_S32 ret = CVIAI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVIAI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_INCAROBJECTDETECTION, argv[1]);
  if (ret != CVIAI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  run(argv[2]);

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();

  return CVIAI_SUCCESS;
}
