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

  cvtdl_model_e enOdModelId =TDL_MODEL_KEYPOINT_FACE_V2;
  cvtdl_handle_t tdl_handle = CVI_TDL_CreateHandle(0);

  ret = CVI_TDL_OpenModel(tdl_handle, enOdModelId, argv[1]);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  cvtdl_image_t image = CVI_TDL_ReadImage(argv[2]);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  cvtdl_face_t obj_meta = {0};
  ret = CVI_TDL_FaceLandmark(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("CVI_TDL_FaceLandmark failed with %#x!\n", ret);
  } else {
    for (int i = 0; i < sizeof(obj_meta.info->landmarks.x); i ++) {
        printf("landmarks id : %d, ", i);
        printf("landmarks x : %f, ", obj_meta.info->landmarks.x[i]);
        printf("landmarks y : %f\n", obj_meta.info->landmarks.y[i]);
    }
  }

  CVI_TDL_ReleaseFaceMeta(&obj_meta);
  CVI_TDL_DestroyImage(image);

exit1:
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
