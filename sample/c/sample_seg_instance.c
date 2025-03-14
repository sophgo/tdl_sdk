#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_fd_model_info(char *model_name, cvtdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "YOLOV8_COCO80") == 0) {
    *model_index = TDL_MODEL_SEG_YOLOV8_COCO80;
  } else if (strcmp(model_name, "FACE_VEHICLE") == 0) {
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

  cvtdl_instance_seg_t inst_seg_meta = {0};
  ret = CVI_TDL_InstanceSegmentation(tdl_handle, enOdModelId, image, &inst_seg_meta);
  if (ret != 0) {
    printf("CVI_TDL_InstanceSegmentation failed with %#x!\n", ret);
  } else {
    if (inst_seg_meta.size <= 0) {
      printf("None to Segmentation\n");
    } else {
        for (int i = 0; i < inst_seg_meta.size; i++) {
            printf("inst_seg_meta_index : %d, ", i);
            printf("box [x1, x2, y1, y2] : %f %f %f %f, ",
              inst_seg_meta.info[i].obj_info->box.x1,
              inst_seg_meta.info[i].obj_info->box.x2,
              inst_seg_meta.info[i].obj_info->box.y1,
              inst_seg_meta.info[i].obj_info->box.y2);
            printf("class : %d, ", inst_seg_meta.info[i].obj_info->class_id);
            printf("score : %f\n", inst_seg_meta.info[i].obj_info->score);
            printf("points=[");
            for (int j = 0; j < inst_seg_meta.info[i].mask_point_size; j++) {
                printf("(%f,%f),", inst_seg_meta.info[i].mask_point[2 * j],
                  inst_seg_meta.info[i].mask_point[2 * j + 1]);
              }
              printf("]\n");
        }
    }
  }

  CVI_TDL_ReleaseInstanceSegMeta(&inst_seg_meta);
  CVI_TDL_DestroyImage(image);

exit1:
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
