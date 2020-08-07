#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cviai.h"
#include "sample_comm.h"
#include "core/utils/vpss_helper.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int genFeatureFile(const char *img_name_list, const char *result_file) {
  FILE *fp;
  if((fp = fopen(img_name_list, "r")) == NULL) {
    printf("file open error!");
    return CVI_FAILURE;
  }

  FILE *fp_feature;
  if((fp_feature = fopen(result_file, "w+")) == NULL) {
    printf("Write file open error!");
    return CVI_FAILURE;
  }

  char line[1024];
  int fail_num = 0;
  int idx = 0;
  while(fscanf(fp, "%[^\n]", line)!=EOF) {
    fgetc(fp);

    printf("%s\n", line);
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frFrame;
    CVI_S32 ret = CVI_AI_ReadImage(line, &blk_fr, &frFrame, PIXEL_FORMAT_RGB_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    int face_count = 0;
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &frFrame, &face, &face_count);
    if (face_count > 0) {
      CVI_AI_FaceQuality(facelib_handle, &frFrame, &face);

      fprintf(fp_feature, "%f\n", face.face_info[0].face_quality.quality);
    }
    if (face_count == 0 || face.face_info[0].face_quality.quality < 0.5 ||
        abs(face.face_info[0].face_quality.pitch) > 0.45 ||
        abs(face.face_info[0].face_quality.yaw) > 0.45) {
      fail_num++;
    }

    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
    printf("num: %d\n", idx);
    idx++;
  }

  printf("fail_num: %d\n", fail_num);
  fclose(fp_feature);
  fclose(fp);

  return CVI_SUCCESS;
}

int main(void) {
  CVI_S32 ret = CVI_SUCCESS;

  ret = MMF_INIT_HELPER(vpssgrp_width, vpssgrp_height, SAMPLE_PIXEL_FORMAT, vpssgrp_width,
                        vpssgrp_height, SAMPLE_PIXEL_FORMAT);
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
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEQUALITY,
                            "/mnt/data/fqnet-v5_shufflenetv2-softmax.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  genFeatureFile("/mnt/data/pic2/list.txt", "/mnt/data/pic2/result.txt");

  CVI_AI_DestroyHandle(facelib_handle);
}
