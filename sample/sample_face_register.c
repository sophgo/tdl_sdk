#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "core/utils/vpss_helper.h"
#include "cviai.h"

cviai_handle_t facelib_handle = NULL;

static CVI_S32 vpssgrp_width = 1920;
static CVI_S32 vpssgrp_height = 1080;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: reg_wider_face <image>.\n");
    printf("image: Face image to register.\n");
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

  VB_BLK blk;
  VIDEO_FRAME_INFO_S frame;
  ret = CVI_AI_ReadImage(argv[1], &blk, &frame, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("Read image failed with %#x!\n", ret);
    return ret;
  }

  int face_count = 0;
  cvai_face_t face;
  memset(&face, 0, sizeof(cvai_face_t));

  CVI_AI_SetSkipVpssPreprocess(facelib_handle, CVI_AI_SUPPORTED_MODEL_RETINAFACE, false);
  CVI_AI_RetinaFace(facelib_handle, &frame, &face, &face_count);
  CVI_AI_FaceAttribute(facelib_handle, &frame, &face);
  printf("face_count %d\n", face.size);

  CVI_AI_Free(&face);
  CVI_VB_ReleaseBlock(blk);
  CVI_AI_DestroyHandle(facelib_handle);
}
