#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "FEATURE_FACE") == 0) {
    *model_index = TDL_MODEL_FEATURE_BMFACER34;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <model name> <model path> <input image path1> <input image path2>\n", argv[0]);
    printf(
      "model name: model name should be one of {"
      "FEATURE_FACE.}}\n");
    printf("model path: Path to feature model.\n");
    printf("input image path: Path to input image1.\n");
    printf("input image path: Path to input image2.\n");
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
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image1 = TDL_ReadImage(argv[3]);
  if (image1 == NULL) {
    printf("read image1 failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_image_t image2 = TDL_ReadImage(argv[4]);
  if (image2 == NULL) {
    printf("read image2 failed with %#x!\n", ret);
    goto exit2;
  }

  tdl_feature_t obj_meta1 = {0}, obj_meta2 = {0};
  ret = TDL_FeatureExtraction(tdl_handle, enOdModelId, image1, &obj_meta1);
  if (ret != 0) {
    printf("TDL_FeatureExtraction failed with %#x!\n", ret);
    goto exit3;
  }

  ret = TDL_FeatureExtraction(tdl_handle, enOdModelId, image2, &obj_meta2);
  if (ret != 0) {
    printf("TDL_FeatureExtraction failed with %#x!\n", ret);
    goto exit3;
  }

  float similarity = 0.0;
  ret = TDL_CaculateSimilarity(obj_meta1, obj_meta2, &similarity);
  printf("similarity is %f\n", similarity);

exit3:
  TDL_ReleaseFeatureMeta(&obj_meta1);
  TDL_ReleaseFeatureMeta(&obj_meta2);
  TDL_DestroyImage(image2);

exit2:
  TDL_DestroyImage(image1);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
