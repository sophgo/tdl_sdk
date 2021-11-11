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

static int run(const char *img_dir, int *count, int *total) {
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
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frame;
    CVI_S32 ret = CVI_AI_ReadImage(line, &blk_fr, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVIAI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &frame, &face);

    face.dms = (cvai_dms_t *)malloc(sizeof(cvai_dms_t));
    face.dms->dms_od.info = NULL;
    CVI_AI_FaceLandmarker(facelib_handle, &frame, &face);
    CVI_AI_YawnClassification(facelib_handle, &frame, &face);
    if (face.dms->yawn_score < 0.75) {
      (*count)++;
    }
    (*total)++;
    CVI_AI_FreeDMS(face.dms);
    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
  }
  closedir(dirp);

  return CVIAI_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    printf(
        "Usage: %s <retinaface model path> <landmark model path> <yawn classifier model path> "
        "<yawn open image dir> <yawn close image dir>.\n",
        argv[0]);
    return CVIAI_FAILURE;
  }

  CVI_AI_PerfettoInit();
  CVI_S32 ret = CVIAI_SUCCESS;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888, 5);
  if (ret != CVIAI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle2(&facelib_handle, 1, 0);
  ret |= CVI_AI_OpenModel(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, argv[1]);
  ret |= CVI_AI_OpenModel(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACELANDMARKER, argv[2]);
  ret |= CVI_AI_OpenModel(facelib_handle, CVI_AI_SUPPORTED_MODEL_YAWNCLASSIFICATION, argv[3]);

  int open = 0, close = 0;
  int open_total = 0, close_total = 0;
  run(argv[4], &open, &open_total);
  run(argv[5], &close, &close_total);

  printf("Num of yawn face -> open: %d/%d, close: %d/%d\n", open_total - open, open_total, close,
         close_total);

  CVI_AI_DestroyHandle(facelib_handle);
}
