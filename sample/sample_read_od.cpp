#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cvi_tdl.h"
//#include "cvi_tdl_media.h"
int ReleaseImage(VIDEO_FRAME_INFO_S *frame) {
  CVI_S32 ret = CVI_SUCCESS;
  if (frame->stVFrame.u64PhyAddr[0] != 0) {
    ret = CVI_SYS_IonFree(frame->stVFrame.u64PhyAddr[0], frame->stVFrame.pu8VirAddr[0]);
    frame->stVFrame.u64PhyAddr[0] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[1] = (CVI_U64)0;
    frame->stVFrame.u64PhyAddr[2] = (CVI_U64)0;
    frame->stVFrame.pu8VirAddr[0] = NULL;
    frame->stVFrame.pu8VirAddr[1] = NULL;
    frame->stVFrame.pu8VirAddr[2] = NULL;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  CVI_S32 ret = 0;
  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_MOBILEDETV2_PEDESTRIAN, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }
  VIDEO_FRAME_INFO_S bg;

  printf("to read image\n");
  if (CVI_SUCCESS != CVI_TDL_LoadBinImage(argv[2], &bg, PIXEL_FORMAT_RGB_888_PLANAR)) {
    printf("cvi_tdl read image failed.");
    CVI_TDL_DestroyHandle(tdl_handle);
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
    cvtdl_object_t obj_meta = {0};
    CVI_TDL_MobileDetV2_Pedestrian(tdl_handle, &bg, &obj_meta);
    // std::stringstream ss;
    // ss << "boxes=[";
    // for (int i = 0; i < obj_meta.size; i++) {
    //   cvtdl_bbox_t b = obj_meta.info[i].bbox;
    //   printf("box=[%.1f,%.1f,%.1f,%.1f]\n", b.x1, b.y1, b.x2, b.y2);

    //   // ss << "[" << obj_meta.info[i].bbox.x1 << "," << obj_meta.info[i].bbox.y1 << ","
    //   //   << obj_meta.info[i].bbox.x2 << "," << obj_meta.info[i].bbox.y2 << "],";
    // }
    // // str_res = ss.str();
    // printf("objsize:%u\n", obj_meta.size);
    CVI_TDL_Free(&obj_meta);
  }
  ReleaseImage(&bg);
  CVI_TDL_DestroyHandle(tdl_handle);

  return ret;
}
