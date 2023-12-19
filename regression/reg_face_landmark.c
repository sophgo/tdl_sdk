#define _GNU_SOURCE
#include <dirent.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_evaluation.h"
#include "cvi_tdl_media.h"
#include "cvi_tdl_perfetto.h"

cvitdl_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

float sqrt3(const float x) {
  union {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
  return u.x;
}

float compute_nme(cvtdl_pts_t *labels, cvtdl_pts_t *pts) {
  float sum = 0.0;
  for (int i = 0; i < 5; ++i) {
    float _dist = 0.0;
    _dist += (float)((labels->x[i] - pts->x[i]) * (labels->x[i] - pts->x[i]));
    _dist += (float)((labels->y[i] - pts->y[i]) * (labels->y[i] - pts->y[i]));
    sum += sqrt3(_dist);
  }
  float _nme = sum / 5;
  float dist = sqrt3((float)((labels->x[0] - labels->x[1]) * (labels->x[0] - labels->x[1]) +
                             (labels->y[0] - labels->y[1]) * (labels->y[0] - labels->y[1])));
  _nme /= dist;
  return _nme;
}

static int run(const char *img_dir, float *nme, int *total) {
  char *line = NULL;
  size_t len = 0;
  ssize_t read;
  float bbox[4];
  float landmark[10];
  char img_name[256];
  char file_name[256];
  sprintf(file_name, "%s/list_test.txt", img_dir);
  FILE *fp = fopen(file_name, "r");
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);
  while ((read = getline(&line, &len, fp)) != -1) {
    memset(bbox, 0, sizeof(bbox));
    memset(landmark, 0, sizeof(landmark));
    memset(img_name, 0, sizeof(img_name));
    char *ptr = strtok(line, " ");

    for (int i = 0; i < 4; ++i) {
      bbox[i] = atof(ptr);
      ptr = strtok(NULL, " ");
    }

    for (int i = 0; i < 10; ++i) {
      landmark[i] = atof(ptr);
      ptr = strtok(NULL, " ");
    }
    strcpy(img_name, ptr);
    img_name[strlen(img_name) - 1] = '\0';
    printf("%s\n", img_name);
    VIDEO_FRAME_INFO_S frame;
    int ret = CVI_TDL_ReadImage(img_handle, img_name, &frame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_TDL_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      continue;
    }

    cvtdl_face_t face;
    memset(&face, 0, sizeof(cvtdl_face_t));
    face.size = 1;
    face.width = frame.stVFrame.u32Width;
    face.height = frame.stVFrame.u32Height;
    face.info = (cvtdl_face_info_t *)malloc(sizeof(cvtdl_face_info_t) * face.size);
    memset(face.info, 0, sizeof(cvtdl_face_info_t) * face.size);
    face.info[0].bbox.x1 = bbox[0];
    face.info[0].bbox.y1 = bbox[2];
    face.info[0].bbox.x2 = bbox[1];
    face.info[0].bbox.y2 = bbox[3];
    face.info[0].pts.size = 5;
    face.info[0].pts.x = (float *)malloc(sizeof(float) * face.info[0].pts.size);
    face.info[0].pts.y = (float *)malloc(sizeof(float) * face.info[0].pts.size);
    for (int i = 0; i < 5; ++i) {
      face.info[0].pts.x[i] = landmark[i];
      face.info[0].pts.y[i] = landmark[i + 1];
    }
    face.dms = (cvtdl_dms_t *)malloc(sizeof(cvtdl_dms_t));
    face.dms->dms_od.info = NULL;
    CVI_TDL_FaceLandmarker(facelib_handle, &frame, &face);
    *nme += compute_nme(&(face.info[0].pts), &(face.dms->landmarks_5));

    (*total)++;
    CVI_TDL_FreeDMS(face.dms);
    CVI_TDL_Free(&face);
    CVI_TDL_ReleaseImage(img_handle, &frame);
  }
  return CVI_TDL_SUCCESS;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <face landmark model path> <face image dir>.\n", argv[0]);
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

  ret = CVI_TDL_OpenModel(facelib_handle, CVI_TDL_SUPPORTED_MODEL_FACELANDMARKER, argv[1]);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  float nme = 0.0;
  int total = 0;
  run(argv[2], &nme, &total);
  printf("nme:%lf total: %d avg: %.3lf\n", nme, total, nme / total);

  CVI_TDL_DestroyHandle(facelib_handle);
}
