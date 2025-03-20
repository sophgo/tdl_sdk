#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_fd_model_info(char *model_name, cvtdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "FACE_VEHICLE") == 0) {
    *model_index = TDL_MODEL_SEG_PERSON_FACE_VEHICLE;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <model name> <model name> <model path> <input image path>\n", argv[0]);
    printf(
      "model name: model name should be one of {"
      "YOLOV8_COCO80, "
      "FACE_VEHICLE.}\n");
    printf("model path: Path to instance seg model.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;

  cvtdl_model_e enOdModelId;
  ret = get_fd_model_info(argv[1], &enOdModelId);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  cvtdl_handle_t tdl_handle = CVI_TDL_CreateHandle(0);

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != 0) {
    printf("open instance seg model failed with %#x!\n", ret);
    goto exit0;
  }

  cvtdl_image_t image = CVI_TDL_ReadImage(argv[3]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  cvtdl_seg_t seg_meta = {0};
  ret = CVI_TDL_SemanticSegmentation(tdl_handle, enOdModelId, image, &seg_meta);
  if (ret != 0) {
    printf("CVI_TDL_InstanceSegmentation failed with %#x!\n", ret);
  } else {
    printf("height : %d, ", seg_meta.height);
    printf("width : %d, ", seg_meta.width);
    printf("output_height : %d, ", seg_meta.output_height);
    printf("output_width : %d\n", seg_meta.output_width);
    for (int x = 0; x < seg_meta.output_height; x ++) {
        for (int y = 0; y < seg_meta.output_width; y ++) {
            printf("%d ", (int)seg_meta.class_id[x * seg_meta.output_width + y]);
        }
        printf("\n");
    }
  }

  CVI_TDL_ReleaseSemanticSegMeta(&seg_meta);
  CVI_TDL_DestroyImage(image);

exit1:
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}