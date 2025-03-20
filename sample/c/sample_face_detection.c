#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "scrfdface") == 0) {
    *model_index = TDL_MODEL_SCRFD_DET_FACE;
  } else if (strcmp(model_name, "retinaface") == 0) {
    *model_index = TDL_MODEL_RETINA_DET_FACE;
  } else if (strcmp(model_name, "retinaface_ir") == 0) {
    *model_index = TDL_MODEL_RETINA_DET_FACE_IR;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <model name> <model path> <input image path> <output image path>\n", argv[0]);
    printf(
        "model name: face detection model name should be one of {scrfdface, "
        "retinaface, "
        "retinaface_ir, "
        "face_mask}.\n");
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

  tdl_face_t obj_meta = {0};
  ret = TDL_FaceDetection(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("face detection failed with %#x!\n", ret);
  } else {
    box_t boxes[obj_meta.size];
    printf("boxes=[");
    for (uint32_t i = 0; i < obj_meta.size; i++) {
      printf("[x1:%f, y1:%f, x2:%f, y2:%f], score:%f, ", obj_meta.info[i].box.x1,
             obj_meta.info[i].box.y1, obj_meta.info[i].box.x2, obj_meta.info[i].box.y2,
             obj_meta.info[i].score);
      boxes[i].x1 = obj_meta.info[i].box.x1;
      boxes[i].y1 = obj_meta.info[i].box.y1;
      boxes[i].x2 = obj_meta.info[i].box.x2;
      boxes[i].y2 = obj_meta.info[i].box.y2;
    }
    printf("]\n");
    TDL_VisualizeRectangle(boxes, obj_meta.size, argv[3], argv[4]);
  }

  TDL_ReleaseFaceMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
