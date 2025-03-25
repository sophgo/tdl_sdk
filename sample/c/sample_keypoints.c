#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "HAND") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_HAND;
  } else if (strcmp(model_name, "LICENSE_PLATE") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strcmp(model_name, "POSE_SIMCC") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_SIMICC;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <model name> <model path> <input image path> <output image path>\n", argv[0]);
    printf(
      "model name: model name should be one of {"
      "HAND, "
      "LICENSE_PLATE.}\n");
    printf("model path: Path to keypoint model.\n");
    printf("input image path: Path to input image.\n");
    printf("output image path: Path to output image.\n");
    return -1;
  }
  int ret = 0;

  tdl_model_e enOdModelId;
  ret = get_model_info(argv[1], &enOdModelId);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image = TDL_ReadImage(argv[3]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_keypoint_t obj_meta = {0};
  ret = TDL_Keypoint(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
      printf("None to detection\n");
    } else {
      point_t point[obj_meta.size];
      for (int i = 0; i < obj_meta.size; i++) {
        printf("obj_meta id : %d, ", i);
        printf("[x, y, score] : %f, %f, %f\n", obj_meta.info[i].x * obj_meta.width,
                                               obj_meta.info[i].y * obj_meta.height,
                                               obj_meta.info[i].score);
        point[i].x = obj_meta.info[i].x * obj_meta.width;
        point[i].y = obj_meta.info[i].y * obj_meta.height;
      }
      TDL_VisualizePoint(point, obj_meta.size, argv[3], argv[4]);
    }
  }

  TDL_ReleaseKeypointMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
