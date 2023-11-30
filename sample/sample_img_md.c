#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cvi_tdl.h"
#if defined(CV181X) || defined(ATHENA2)
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif

int main(int argc, char *argv[]) {
  cvitdl_handle_t tdl_handle = NULL;
  printf("start to run img md\n");
  CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();

  VIDEO_FRAME_INFO_S bg, frame;
// printf("toread image:%s\n",argv[1]);
#ifdef SIMPLY_MODEL
  IVE_IMAGE_S image1;
#else
  const char *strf1 = "/mnt/data/admin1_data/alios_test/set/a.jpg";
  IVE_IMAGE_S image1 = CVI_IVE_ReadImage(ive_handle, strf1, IVE_IMAGE_TYPE_U8C1);
#endif
  ret = CVI_SUCCESS;

#if defined(CV181X) || defined(ATHENA2)
  int imgw = image1.u32Width;
#else
  int imgw = image1.u16Width;
#endif

  if (imgw == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }
#if defined(CV181X) || defined(ATHENA2)
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg, false);
#endif

  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }
#ifdef SIMPLY_MODEL
  IVE_IMAGE_S image2;
#else
  const char *strf2 = "/mnt/data/admin1_data/alios_test/set/b.jpg";
  IVE_IMAGE_S image2 = CVI_IVE_ReadImage(ive_handle, strf2, IVE_IMAGE_TYPE_U8C1);
#endif
  ret = CVI_SUCCESS;
#if defined(CV181X) || defined(ATHENA2)
  int imgw2 = image2.u32Width;
#else
  int imgw2 = image2.u16Width;
#endif
  if (imgw2 == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }

#if defined(CV181X) || defined(ATHENA2)
  ret = CVI_IVE_Image2VideoFrameInfo(&image2, &frame);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image2, &frame, false);
#endif

  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }
  cvtdl_object_t obj_meta;
  CVI_TDL_Set_MotionDetection_Background(tdl_handle, &bg);

  CVI_TDL_MotionDetection(tdl_handle, &frame, &obj_meta, 20, 50);
  // CVI_TDL_MotionDetection(tdl_handle, &frame, &obj_meta, 20, 50);
  // CVI_TDL_MotionDetection(tdl_handle, &frame, &obj_meta, 20, 50);
  // VIDEO_FRAME_INFO_S motionmap;
  // ret = CVI_TDL_GetMotionMap(tdl_handle, &motionmap);

#ifndef SIMPLY_MODEL
  CVI_TDL_DumpImage("img1.bin", &bg);
  CVI_TDL_DumpImage("img2.bin", &frame);
#endif
  // CVI_TDL_DumpImage("md.bin", &motionmap);

  for (int i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f]\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
  }

  CVI_TDL_Free(&obj_meta);
  CVI_SYS_FreeI(ive_handle, &image1);
  CVI_SYS_FreeI(ive_handle, &image2);

  CVI_TDL_DestroyHandle(tdl_handle);
  CVI_IVE_DestroyHandle(ive_handle);
  return ret;
}
