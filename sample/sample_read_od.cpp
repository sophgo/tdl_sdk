#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cviai.h"
#include "evaluation/cviai_media.h"

int main(int argc, char *argv[]) {
  CVI_S32 ret = 0;
  cviai_handle_t ai_handle = NULL;
  ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_AI_OpenModel(ai_handle, CVI_AI_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  VIDEO_FRAME_INFO_S bg;

  printf("to read image\n");
  if (CVI_SUCCESS != CVI_AI_ReadImage(argv[2], &bg, PIXEL_FORMAT_RGB_888_PLANAR)) {
    printf("cviai read image failed.");
    CVI_AI_DestroyHandle(ai_handle);
    return -1;
  }
  int loop_count = 1;
  if (argc == 4) {
    loop_count = atoi(argv[3]);
  }
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
  }

  for (int i = 0; i < loop_count; i++) {
    cvai_object_t obj_meta = {0};
    CVI_AI_MobileDetV2_Pedestrian(ai_handle, &bg, &obj_meta);
    // std::stringstream ss;
    // ss << "boxes=[";
    // for (int i = 0; i < obj_meta.size; i++) {
    //   cvai_bbox_t b = obj_meta.info[i].bbox;
    //   printf("box=[%.1f,%.1f,%.1f,%.1f]\n", b.x1, b.y1, b.x2, b.y2);

    //   // ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
    //   //   << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
    // }
    // // str_res = ss.str();
    // printf("objsize:%u\n", obj_meta.size);
    CVI_AI_Free(&obj_meta);
  }
  CVI_AI_ReleaseImage(&bg);
  CVI_AI_DestroyHandle(ai_handle);

  return ret;
}
