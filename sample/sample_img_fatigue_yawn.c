#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"

int process_image_file(cvitdl_handle_t tdl_handle, char *imgf, cvtdl_face_t *p_obj) {
  VIDEO_FRAME_INFO_S bg;
  imgprocess_t img_handle;
  CVI_TDL_Create_ImageProcessor(&img_handle);

  int ret = CVI_TDL_ReadImage(img_handle, imgf, &bg, PIXEL_FORMAT_RGB_888);
  if (ret != CVI_SUCCESS) {
    printf("open img failed with %#x!\n", ret);
    return ret;
  } else {
    printf("image read,width:%d\n", bg.stVFrame.u32Width);
    printf("image read,hidth:%d\n", bg.stVFrame.u32Height);
  }

  ret = CVI_TDL_FaceDetection(tdl_handle, &bg, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, p_obj);

  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_ScrFDFace failed with %#x!\n", ret);
    return ret;
  }

  if (p_obj->size > 0) {
    ret = CVI_TDL_FaceLandmarker(tdl_handle, &bg, p_obj);
    if (ret != CVI_SUCCESS) {
      printf("CVI_TDL_FaceLandmarker failed with %#x!\n", ret);
      return ret;
    }
  } else {
    printf("cannot find face_meta\n");
  }

  p_obj->dms = (cvtdl_dms_t *)malloc(sizeof(cvtdl_dms_t));
  p_obj->dms->dms_od.info = NULL;

  ret = CVI_TDL_YawnClassification(tdl_handle, &bg, p_obj);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_SUPPORTED_MODEL_YAWNCLASSIFICATION failed with %#x!\n", ret);
    return ret;
  }

  CVI_TDL_ReleaseImage(img_handle, &bg);
  CVI_TDL_Destroy_ImageProcessor(img_handle);
  return ret;
}

int main(int argc, char *argv[]) {
  int vpssgrp_width = 1920;
  int vpssgrp_height = 1080;
  if (argc != 5) {
    printf(
        "Usage: %s <face detection model path> <face landmarker model path> <yawn classification "
        "model path> <input image path>\n",
        argv[0]);
    printf("face detection model path: Path to face detection model cvimodel.\n");
    printf("face landmarker model path: Path to face landmarker model cvimodel.\n");
    printf("yawn classification model path: Path to yawn classification model cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return CVI_FAILURE;
  }
  CVI_S32 ret = MMF_INIT_HELPER2(vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1,
                                 vpssgrp_width, vpssgrp_height, PIXEL_FORMAT_RGB_888, 1);
  if (ret != CVI_TDL_SUCCESS) {
    printf("Init sys failed with %#x!\n", ret);
    return ret;
  }

  cvitdl_handle_t tdl_handle = NULL;
  ret = CVI_TDL_CreateHandle(&tdl_handle);
  if (ret != CVI_SUCCESS) {
    printf("Create tdl handle failed with %#x!\n", ret);
    return ret;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_SCRFDFACE, argv[1]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_TDL_SUPPORTED_MODEL_SCRFDFACE model failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_FACELANDMARKER, argv[2]);
  if (ret != CVI_SUCCESS) {
    printf("open CVI_TDL_SUPPORTED_MODEL_FACELANDMARKER model failed with %#x!\n", ret);
    return ret;
  }
  ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YAWNCLASSIFICATION, argv[3]);
  if (ret != CVI_SUCCESS) {
    printf("CVI_TDL_SUPPORTED_MODEL_YAWNCLASSIFICATION model failed with %#x!\n", ret);
    return ret;
  }

  cvtdl_face_t face_meta = {0};
  ret = process_image_file(tdl_handle, argv[4], &face_meta);
  if (ret != CVI_SUCCESS) {
    printf("Process image failed with %#x!\n", ret);
    return ret;
  }

  for (uint32_t i = 0; i < face_meta.size; i++) {
    printf("[%f,%f,%f,%f],", face_meta.info[i].bbox.x1, face_meta.info[i].bbox.y1,
           face_meta.info[i].bbox.x2, face_meta.info[i].bbox.y2);
    printf("face %d yawn score :%f\n", i, face_meta.dms[i].yawn_score);
  }

  CVI_TDL_Free(&face_meta);
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
