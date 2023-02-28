#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cviai.h"
#ifdef CV181X
#include <cvi_ive.h>
#else
#include "ive/ive.h"
#endif

int main(int argc, char *argv[]) {
  cviai_handle_t ai_handle = NULL;
  printf("start to run img md\n");
  CVI_S32 ret = CVI_AI_CreateHandle(&ai_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create ai handle failed with %#x!\n", ret);
    return ret;
  }
  IVE_HANDLE ive_handle = CVI_IVE_CreateHandle();
  const char *strf1 = "/mnt/data/admin1_data/alios_test/set/a.jpg";
  const char *strf2 = "/mnt/data/admin1_data/alios_test/set/b.jpg";

  VIDEO_FRAME_INFO_S bg, frame;
  // printf("toread image:%s\n",argv[1]);
  IVE_IMAGE_S image1 = CVI_IVE_ReadImage(ive_handle, strf1, IVE_IMAGE_TYPE_U8C1);
  ret = CVI_SUCCESS;

#ifdef CV181X
  int imgw = image1.u32Width;
#else
  int imgw = image1.u16Width;
#endif

  if (imgw == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }
#ifdef CV181X
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image1, &bg, false);
#endif

  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }
  IVE_IMAGE_S image2 = CVI_IVE_ReadImage(ive_handle, strf2, IVE_IMAGE_TYPE_U8C1);
  ret = CVI_SUCCESS;
#ifdef CV181X
  int imgw2 = image2.u32Width;
#else
  int imgw2 = image2.u16Width;
#endif
  if (imgw2 == 0) {
    printf("Read image failed with %x!\n", ret);
    return CVI_FAILURE;
  }

#ifdef CV181X
  ret = CVI_IVE_Image2VideoFrameInfo(&image2, &frame);
#else
  ret = CVI_IVE_Image2VideoFrameInfo(&image2, &frame, false);
#endif

  if (ret != CVI_SUCCESS) {
    printf("Convert to video frame failed with %#x!\n", ret);
    return ret;
  }
  cvai_object_t obj_meta;
  CVI_AI_Set_MotionDetection_Background(ai_handle, &bg);
  CVI_AI_Set_MotionDetection_ROI(ai_handle, 0, 0, 512, 512);

  CVI_AI_MotionDetection(ai_handle, &frame, &obj_meta, 20, 50);
  CVI_AI_MotionDetection(ai_handle, &frame, &obj_meta, 20, 50);
  CVI_AI_MotionDetection(ai_handle, &frame, &obj_meta, 20, 50);
  // VIDEO_FRAME_INFO_S motionmap;
  // ret = CVI_AI_GetMotionMap(ai_handle, &motionmap);

  CVI_AI_DumpImage("img1.bin", &bg);
  CVI_AI_DumpImage("img2.bin", &frame);
  // CVI_AI_DumpImage("md.bin", &motionmap);

  for (int i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f]\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1,
           obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2);
  }

  CVI_AI_Free(&obj_meta);
  CVI_SYS_FreeI(ive_handle, &image1);
  CVI_SYS_FreeI(ive_handle, &image2);

  CVI_AI_DestroyHandle(ai_handle);
  CVI_IVE_DestroyHandle(ive_handle);
  return ret;
}
