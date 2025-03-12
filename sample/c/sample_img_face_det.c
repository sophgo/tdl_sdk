#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_fd_model_info(char *model_name, cvtdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "scrfdface") == 0) {
    *model_index = TDL_MODEL_SCRFDFACE;
  } else if (strcmp(model_name, "retinaface") == 0) {
    *model_index = TDL_MODEL_RETINAFACE;
  } else if (strcmp(model_name, "retinaface_ir") == 0) {
    *model_index = TDL_MODEL_RETINAFACE_IR;
  } else if (strcmp(model_name, "face_mask") == 0) {
    *model_index = TDL_MODEL_MASKFACERECOGNITION;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf(
        "model name: face detection model name should be one of {scrfdface, "
        "retinaface, "
        "retinaface_ir, "
        "face_mask}.\n");
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;
  cvtdl_handle_t tdl_handle = CVI_TDL_CreateHandle(0);

  cvtdl_model_e enOdModelId;
  if (get_fd_model_info(argv[1], &enOdModelId) == -1) {
    printf("unsupported model: %s\n", argv[1]);
    return -1;
  }

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    return ret;
  }

  printf("enter path = %s\n", argv[3]);

  cvtdl_image_t image = CVI_TDL_ReadImage(argv[3]);
  printf("enetr read success, image is NULL? %d\n", image==NULL);
  cvtdl_face_t obj_meta = {0};
  ret = CVI_TDL_InitFaceMeta(&obj_meta, 1, 0);
  ret = CVI_TDL_FaceDetection(tdl_handle, enOdModelId, image, &obj_meta);
  printf("boxes=[");
  for (uint32_t i = 0; i < obj_meta.size; i++) {
    printf("[%f,%f,%f,%f],", obj_meta.info[i].box.x1, obj_meta.info[i].box.y1,
           obj_meta.info[i].box.x2, obj_meta.info[i].box.y2);
  }
  printf("]\n");
  CVI_TDL_ReleaseFaceMeta(&obj_meta);
  CVI_TDL_DestroyImage(image);
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
