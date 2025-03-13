#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"


int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <model path> <input image path>\n", argv[0]);
    printf("model path: Path to cvimodel.\n");
    printf("input image path: Path to input image.\n");
    return -1;
  }
  int ret = 0;

  cvtdl_model_e enOdModelId = TDL_MODEL_ATTRIBUTE_FACE;
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
  ret = CVI_TDL_FaceAttribute(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("CVI_TDL_FaceAttribute failed with %#x!\n", ret);
  } else {
    printf("gender score:%f,age score:%f,glass score:%f,mask score:%f\n",
        obj_meta.info->gender_score, obj_meta.info->age,
        obj_meta.info->glass_score, obj_meta.info->mask_score);
    printf("Gender:%s\n", obj_meta.info->age > 0.5 ? "Male" : "Female");
    printf("Age:%d\n", obj_meta.info->age * 100);
    printf("Glass:%s\n", obj_meta.info->glass_score > 0.5 ? "Yes" : "No");
    printf("Mask:%s\n", obj_meta.info->mask_score > 0.5 ? "Yes" : "No");
  }

  CVI_TDL_ReleaseFaceMeta(&obj_meta);
  CVI_TDL_DestroyImage(image);

exit1:
  CVI_TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  CVI_TDL_DestroyHandle(tdl_handle);
  return ret;
}
