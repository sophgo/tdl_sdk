#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

#define AUDIOFORMATSIZE 2

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "CLS_SOUND_BABAY_CRY") == 0) {
    *model_index = TDL_MODEL_CLS_BABAY_CRY;
  } else if (strcmp(model_name, "CLS_SOUND_COMMAND") == 0) {
    *model_index = TDL_MODEL_CLS_SOUND_COMMAND;
  } else if (strcmp(model_name, "CLS_RGBLIVENESS") == 0) {
    *model_index = TDL_MODEL_CLS_RGBLIVENESS;
  } else {
    ret = -1;
  }
  return ret;
}

int main(int argc, char *argv[]) {
  if (argc != 4 && argc != 6) {
    printf("Usage: %s <model name> <model path> <input image path>\n", argv[0]);
    printf("Usage: %s <model name> <sound_model_path> <bin_data_path> <sample_rate> <seconds>\n", argv[0]);
    printf("model path: Path to cvimodel.\n");
    printf("bin_data_path: Path to input image.\n");
    printf("sample_rate: Path to input image.\n");
    printf("seconds: Path to input image.\n");
    printf(
      "model name: model name should be one of {"
        "CLS_SOUND_BABAY_CRY, "
        "CLS_SOUND_COMMAND, "
        "CLS_RGBLIVENESS}.\n");
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

  tdl_image_t image;

  if (argc == 4) {
    image = TDL_ReadImage(argv[3]);
    if (image == NULL) {
      printf("read image failed with %#x!\n", ret);
      goto exit1;
    }
  } else {
    const char *bin_data_path = argv[3];
    int sample_rate = atoi(argv[4]);
    int seconds = atoi(argv[5]);
    int size = sample_rate * AUDIOFORMATSIZE * seconds;

    image = TDL_ReadAudio(bin_data_path, size);
    if (image == NULL) {
      printf("read image failed with %#x!\n", ret);
      goto exit1;
    }

  }
  tdl_class_info_t obj_info = {0};

  ret = TDL_Classfification(tdl_handle, enOdModelId, image, &obj_info);

  if (ret != 0) {
    printf("TDL_Classfification failed with %#x!\n", ret);
  } else {
    printf("pred_label: %d, score = %f\n", obj_info.class_id, obj_info.score);
  }

  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
