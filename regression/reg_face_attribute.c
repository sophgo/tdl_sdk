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
#include "utils/vpss_helper.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

static int genFeatureFile(const char *img_name_list, const char *feature_dir) {
  FILE *fp;
  if((fp = fopen(img_name_list, "r")) == NULL) {
    printf("file open error!");
    return CVI_FAILURE;
  }

  char line[1024];
  int file_idx = 0;
  while(fscanf(fp, "%[^\n]", line)!=EOF) {
    fgetc(fp);

    printf("%s\n", line);
    VB_BLK blk_fr;
    VIDEO_FRAME_INFO_S frFrame;
    CVI_S32 ret = CVI_AI_ReadImage(line, &blk_fr, &frFrame, PIXEL_FORMAT_BGR_888);
    if (ret != CVI_SUCCESS) {
      printf("Read image failed with %#x!\n", ret);
      return ret;
    }

    int face_count = 0;
    cvai_face_t face;
    memset(&face, 0, sizeof(cvai_face_t));
    CVI_AI_RetinaFace(facelib_handle, &frFrame, &face, &face_count);
    CVI_AI_FaceAttribute(facelib_handle, &frFrame, &face);

    char base_name[150] = "\0";
    char temp[9];
    sprintf(temp, "%d.txt", file_idx);
    strcat(base_name, feature_dir);
    strcat(base_name, temp);

    FILE *fp_feature;
    if((fp_feature = fopen(base_name, "w+")) == NULL) {
      printf("Write file open error!");
      return CVI_FAILURE;
    }
    for (int i = 0; i < face.face_info[0].face_feature.size; i++) {
      fprintf(fp_feature, "%d\n", (int)face.face_info[0].face_feature.ptr[i]);
    }
    fclose(fp_feature);

    CVI_AI_Free(&face);
    CVI_VB_ReleaseBlock(blk_fr);
    file_idx++;
  }
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
  ret = CVI_AI_SetModelPath(facelib_handle, CVI_AI_SUPPORTED_MODEL_FACEATTRIBUTE,
                            "/mnt/data/bmface.cvimodel");
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);

  genFeatureFile("/mnt/data/probe_name.txt", "/mnt/data/probe_feature/");
  genFeatureFile("/mnt/data/gallery_name.txt", "/mnt/data/gallery_feature/");

  CVI_AI_DestroyHandle(facelib_handle);
}
