#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_name, tdl_model_e *model_index_d,  tdl_model_e *model_index_k) {
  int ret = 0;
  if (strcmp(model_name, "HAND") == 0) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_HAND;
    *model_index_k = TDL_MODEL_KEYPOINT_HAND;
  } else if (strcmp(model_name, "LICENSE_PLATE") == 0) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
    *model_index_k = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strcmp(model_name, "POSE_SIMCC") == 0) {
    *model_index_d = TDL_MODEL_MBV2_DET_PERSON;
    *model_index_k = TDL_MODEL_KEYPOINT_SIMICC;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <model name> <model path> <model path> <input image path>\n", argv[0]);
    printf(
      "model name: model name should be one of {"
      "HAND, "
      "LICENSE_PLATE"
      "POSE_SIMCC.}\n");
    printf("model path: Path to detect model.\n");
    printf("model path: Path to keypoint model.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;

  tdl_model_e enOdModelId_d, enOdModelId_k;
  ret = get_model_info(argv[1], &enOdModelId_d, &enOdModelId_k);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId_d, argv[2]);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, enOdModelId_k, argv[3]);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_image_t image = TDL_ReadImage(argv[4]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit2;
  }

  tdl_object_t obj_meta = {0};

  ret = TDL_Detection(tdl_handle, enOdModelId_d, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
    goto exit3;
  } else if (obj_meta.size <= 0){
      printf("None to detection\n");
      goto exit3;
  }

  ret = TDL_DetectionKeypoint(tdl_handle, enOdModelId_k, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    for (int i = 0; i < obj_meta.size; i++) {
      for (int j = 0; j < obj_meta.info[0].landmark_size; j++) {
          printf("obj_meta id: %d, ", i);
          printf("[x, y]: %f, %f\n", obj_meta.info[i].landmark_properity[j].x * obj_meta.width,
                                     obj_meta.info[i].landmark_properity[j].y * obj_meta.height);
      }
    }
  }

exit3:
  TDL_ReleaseObjectMeta(&obj_meta);
  TDL_DestroyImage(image);

exit2:
  TDL_CloseModel(tdl_handle, enOdModelId_k);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId_d);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
