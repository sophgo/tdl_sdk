#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <cvimath/cvimath.h>

#include "cviai.h"
#include "core/utils/vpss_helper.h"

#define NUM_DATASET       6000
#define RESULT_FILE_PATH   "/mnt/data/lfw_result.txt"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int getMaxFace(const cvai_face_t *face)
{
  int face_idx = 0;
  float max_area = 0;
  for (int i = 0; i < face->size; i++) {
    cvai_bbox_t bbox = face->info[i].bbox;
    float curr_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
    if (curr_area > max_area) {
      max_area = curr_area;
      face_idx = i;
    }
  }

  return face_idx;
}

static float evalDifference(const cvai_feature_t *features1, const cvai_feature_t *features2)
{
  int32_t value1 = 0, value2 = 0, value3 = 0;
  for (uint32_t i = 0; i < features1->size; i++) {
    value1 += (short)features1->ptr[i] * features1->ptr[i];
    value2 += (short)features2->ptr[i] * features2->ptr[i];
    value3 += (short)features1->ptr[i] * features2->ptr[i];
  }

  return 1 - ((float)value3 / (sqrt((double)value1) * sqrt((double)value2)));
}

static int genScore(int *eval_label, float *eval_score, const char *pair_path) {
  FILE *fp;
  if((fp = fopen(pair_path, "r")) == NULL) {
    printf("file open error: %s!\n", pair_path);
    return CVI_FAILURE;
  }

  char line[1024];
  int i = 0;
  while(fscanf(fp, "%[^\n]", line)!=EOF) {
    fgetc(fp);

    const char* delim = " ";
    char* label = strtok(line, delim);
    printf("%s\n", label);
    char* name1 = strtok(NULL, delim);
    printf("%s\n", name1);
    char* name2 = strtok(NULL, delim);
    printf("%s\n", name2);

    VB_BLK blk1;
    VIDEO_FRAME_INFO_S frame1;
    CVI_S32 ret = CVI_AI_ReadImage(name1, &blk1, &frame1, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image1 failed with %#x!\n", ret);
      return ret;
    }

    VB_BLK blk2;
    VIDEO_FRAME_INFO_S frame2;
    ret = CVI_AI_ReadImage(name2, &blk2, &frame2, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image2 failed with %#x!\n", ret);
      return ret;
    }

    int face_count = 0;
    cvai_face_t face1, face2;
    memset(&face1, 0, sizeof(cvai_face_t));
    memset(&face2, 0, sizeof(cvai_face_t));

    CVI_AI_RetinaFace(facelib_handle, &frame1, &face1, &face_count);
    CVI_AI_RetinaFace(facelib_handle, &frame2, &face2, &face_count);

    CVI_AI_FaceAttribute(facelib_handle, &frame1, &face1);
    CVI_AI_FaceAttribute(facelib_handle, &frame2, &face2);

    int face_idx1 = getMaxFace(&face1);
    int face_idx2 = getMaxFace(&face2);

    float feature_diff = evalDifference(&face1.info[face_idx1].face_feature,
                                        &face2.info[face_idx2].face_feature);
    float score = 1.0 - (0.5 * feature_diff);

    eval_label[i] = atoi(label);
    eval_score[i] = score;

    CVI_AI_Free(&face1);
    CVI_AI_Free(&face2);
    CVI_VB_ReleaseBlock(blk1);
    CVI_VB_ReleaseBlock(blk2);
    i++;
  }
  fclose(fp);

  return CVI_SUCCESS;
}

static int compare(const void *arg1, const void *arg2) {
  return  (*(float *)arg1 < *(float *)arg2);
}

static void evalAUC(int *y, float *pred)
{
  int pos = 0;
  int neg = 0;
  float pred_sort[NUM_DATASET];
  float max_value = 0;
  for (int i = 0; i < NUM_DATASET; i++) {
    if (y[i] == 1) pos++;
    if (y[i] == 0) neg++;
    pred_sort[i] = pred[i];
    if (max_value < pred[i]) max_value = pred[i];
  }

  qsort((void *)pred_sort, NUM_DATASET, sizeof(float), compare);
  printf("%f, %f\n", pred_sort[0], max_value);

  FILE *fp;
  if((fp = fopen(RESULT_FILE_PATH, "w")) == NULL) {
    printf("LFW result file open error!");
    return;
  }

  for (int i = 0; i < NUM_DATASET; i++) {
    float thr = pred_sort[i];

    int tpn = 0, fpn = 0;
    for (int j = 0; j < NUM_DATASET; j++) {
      if (pred[j] >= thr) {
        if (y[j] == 1) {
          tpn++;
        } else {
          fpn++;
        }
      }
    }

    float tpr = (float)tpn / pos;
    float fpr = (float)fpn / neg;
    fprintf(fp, "thr: %f, tpr: %f, fpr: %f\n", thr, tpr, fpr);
  }

  fclose(fp);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: reg_lfw <pair_txt_path>.\n");
    printf("Pair txt format: lable image1_path image2_path.\n");
    return CVI_FAILURE;
  }

  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, vpssgrp_width,
                        vpssgrp_height, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_CreateHandle(&facelib_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE,
                            "/mnt/data/retina_face.cvimodel");
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                            "/mnt/data/bmface.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  int eval_label[NUM_DATASET];
  float eval_score[NUM_DATASET];
  genScore(eval_label, eval_score, argv[1]);
  evalAUC(eval_label, eval_score);

  CVI_AI_DestroyHandle(facelib_handle);
  CVI_SYS_Exit();
}
