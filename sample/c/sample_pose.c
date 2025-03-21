#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;

  tdl_model_e enOdModelId = TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17;
  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId, argv[1]);
  if (ret != 0) {
    printf("open pose model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image = TDL_ReadImage(argv[2]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_object_t obj_meta = {0};
  ret = TDL_Detection(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Pose failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
        printf("None to detection\n");
    } else {
        for (int i = 0; i < obj_meta.size; i++) {
            printf("obj_meta_index : %d, ", i);
            printf("class_id : %d, ", obj_meta.info[i].class_id);
            printf("score : %f, ", obj_meta.info[i].score);
            printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                                             obj_meta.info[i].box.x2,
                                             obj_meta.info[i].box.y1,
                                             obj_meta.info[i].box.y2);
            for (int j = 0; j < 17; j ++) {
                printf("pose : %d: %f %f %f\n", j, obj_meta.info[i].landmark_properity[j].x,
                    obj_meta.info[i].landmark_properity[j].y,
                    obj_meta.info[i].landmark_properity[j].score);
            }
        }
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
