#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "YOLOV10_COCO80") == 0) {
    *model_index = TDL_MODEL_YOLOV10_DET_COCO80;
  } else if (strcmp(model_name, "YOLOV8N_HEAD_HARDHAT") == 0) {
    *model_index = TDL_MODEL_YOLOV8N_DET_HEAD_HARDHAT;
  } else if (strcmp(model_name, "YOLOV8N_PERSON_VEHICLE") == 0) {
    *model_index = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  } else if (strcmp(model_name, "KEYPOINT_FACE_V2") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_FACE_V2;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <model name> <model path> <input image path> <output image path>\n", argv[0]);
    printf(
        "model name: obj detection model name should be one of {"
        "YOLOV10_COCO80, "
        "YOLOV8N_HEAD_HARDHAT, "
        "YOLOV8N_PERSON_VEHICLE, "
        "KEYPOINT_FACE_V2}.\n");
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    printf("output image path: Path to output image.\n");
    return -1;
  }
  int ret = 0;

  tdl_model_e enOdModelId;
  if (get_model_info(argv[1], &enOdModelId) == -1) {
    printf("unsupported model: %s\n", argv[1]);
    return -1;
  }

  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId, argv[2]);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image = TDL_ReadImage(argv[3]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_object_t obj_meta = {0};
  ret = TDL_Detection(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
      printf("None to detection\n");
    } else {
      box_t boxes[obj_meta.size];
      for (int i = 0; i < obj_meta.size; i++) {
        printf("obj_meta_index : %d, ", i);
        printf("class_id : %d, ", obj_meta.info[i].class_id);
        printf("score : %f, ", obj_meta.info[i].score);
        printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                                          obj_meta.info[i].box.x2,
                                          obj_meta.info[i].box.y1,
                                          obj_meta.info[i].box.y2);
        boxes[i].x1 = obj_meta.info[i].box.x1;
        boxes[i].y1 = obj_meta.info[i].box.y1;
        boxes[i].x2 = obj_meta.info[i].box.x2;
        boxes[i].y2 = obj_meta.info[i].box.y2;
      }
      TDL_VisualizeRectangle(boxes, obj_meta.size, argv[3], argv[4]);
    }
  }

  TDL_ReleaseObjectMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
