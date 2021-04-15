#define _GNU_SOURCE
#include "core/utils/vpss_helper.h"
#include "cviai.h"
#include "ive/ive.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <detection_model_path> <alphapose_model_path> <video_frames_folder>.\n",
           argv[0]);
    return CVI_FAILURE;
  }
  CVI_S32 ret = CVI_SUCCESS;

  // Init VB pool size.
  const CVI_S32 vpssgrp_width = 1920;
  const CVI_S32 vpssgrp_height = 1080;
  ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 5, vpssgrp_width,
                         vpssgrp_height, PIXEL_FORMAT_RGB_888_PLANAR, 5);
  if (ret != CVI_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  // Init cviai handle.
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create handle failed with %#x!\n", ret);
    return ret;
  }

  // Setup model path and model config.
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("Set model retinaface failed with %#x!\n", ret);
    return ret;
  }
  CVI_AI_SetSkipVpssPreprocess(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_D0, false);
  ret = CVI_AI_SetModelPath(ai_handle, CVI_AI_SUPPORTED_MODEL_ALPHAPOSE, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("Set model alphapose failed with %#x!\n", ret);
    return ret;
  }

  struct dirent **entry_list;
  int count;
  int i;

  count = scandir(argv[3], &entry_list, 0, alphasort);
  if (count < 0) {
    perror("scandir");
    return EXIT_FAILURE;
  }

  VB_BLK blk1;
  VIDEO_FRAME_INFO_S fdFrame;
  cvai_object_t obj;
  memset(&obj, 0, sizeof(cvai_object_t));
  for (i = 0; i < count; i++) {
    struct dirent *dp;

    dp = entry_list[i];
    if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) {
      // skip . & ..
    } else {
      int image_path_len = strlen(argv[3]) + strlen(dp->d_name) + 2;
      // printf("%s/%s\n", argv[3], dp->d_name);
      char image_path[image_path_len];
      memset(image_path, '\0', image_path_len);
      // printf("path1 : %s\n", argv[3]);
      // printf("path2 : %s\n", dp->d_name);
      strcat(image_path, argv[3]);
      strcat(image_path, "/");
      strcat(image_path, dp->d_name);

      ret = CVI_AI_ReadImage(image_path, &blk1, &fdFrame, PIXEL_FORMAT_RGB_888);
      if (ret != CVI_SUCCESS) {
        printf("Read image1 failed with %#x!\n", ret);
        return ret;
      }
      printf("\nRead image : %s ", image_path);

      // Run inference and print result.
      CVI_AI_MobileDetV2_D0(ai_handle, &fdFrame, &obj, CVI_DET_TYPE_PEOPLE);
      printf("; People found %x ", obj.size);

      CVI_AI_AlphaPose(ai_handle, &fdFrame, &obj);

      CVI_AI_Fall(ai_handle, &obj);
      if (obj.size > 0 && obj.info[0].pedestrian_properity != NULL) {
        printf("; fall score %d ", obj.info[0].pedestrian_properity->fall);
      }

      CVI_VB_ReleaseBlock(blk1);
      CVI_AI_Free(&obj);

      free(dp);
    }
  }
  free(entry_list);
  // Free image and handles.
  // CVI_SYS_FreeI(ive_handle, &image);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}