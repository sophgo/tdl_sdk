#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

#define AUDIOFORMATSIZE 2

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "cls_sound_babay_cry") != NULL) {
    *model_index = TDL_MODEL_CLS_SOUND_BABAY_CRY;
  } else if (strstr(model_path, "cls_rgbliveness") != NULL) {
    *model_index = TDL_MODEL_CLS_RGBLIVENESS;
  } else if (strstr(model_path, "cls_sound_dakaiqianlu") != NULL ||
             strstr(model_path, "cls_sound_nihaoshiyun") != NULL ||
             strstr(model_path, "cls_sound_xiaoaixiaoai") != NULL) {
    *model_index = TDL_MODEL_CLS_SOUND_COMMAND;
  } else if (strstr(model_path, "cls_hand_gesture_128_128") != NULL) {
    *model_index = TDL_MODEL_CLS_HAND_GESTURE;
  } else if (strstr(model_path, "cls_keypoint_hand_gesture") != NULL) {
    *model_index = TDL_MODEL_CLS_KEYPOINT_HAND_GESTURE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  Image processing mode:\n");
  printf("    %s -m <model_path> -i <input_image>\n", prog_name);
  printf("    %s --model_path <path> --input <image>\n", prog_name);
  printf("  Bin processing mode:\n");
  printf("    %s -m <model_path> -b <input_bin> -r <rate> -s <time>\n",
         prog_name);
  printf("    %s -m <model_path> -b <input_bin> -c <data count>\n", prog_name);
  printf(
      "    %s --model_path <path> --bin_data <input_bin> --sample-rate <rate> "
      "--seconds <time>\n",
      prog_name);
  printf(
      "    %s --model_path <path> --bin_data <input_bin> --count <data "
      "count>\n",
      prog_name);
  printf("\nOptions:\n");
  printf(
      "  -m, --model_path     Path to model, "
      "<cls_sound_babay_cry_xxx>"
      "<cls_rgbliveness_xxx>\n");
  printf("  -i, --input          Path to input image (image mode)\n");
  printf("  -b, --bin_data       Path to input data (bin mode)\n");
  printf("  -r, --sample-rate    Sample rate in Hz (bin mode)\n");
  printf("  -s, --seconds        Duration in seconds (bin mode)\n");
  printf("  -c, --count          Data count (bin mode)\n");
  printf("  -h, --help           Display this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image = NULL;
  char *bin_data = NULL;
  char *sample_rate = NULL;
  char *seconds = NULL;
  char *data_count = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input", required_argument, 0, 'i'},
                                  {"bin_data", required_argument, 0, 'b'},
                                  {"sample-rate", required_argument, 0, 'r'},
                                  {"seconds", required_argument, 0, 's'},
                                  {"count", required_argument, 0, 'c'},
                                  {"help", no_argument, 0, 'h'},
                                  {0, 0, 0, 0}};

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "m:i:b:r:s:c:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'i':
        input_image = optarg;
        break;
      case 'b':
        bin_data = optarg;
        break;
      case 'r':
        sample_rate = optarg;
        break;
      case 's':
        seconds = optarg;
        break;
      case 'c':
        data_count = optarg;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
        // getopt_long 已经打印了错误信息
        print_usage(argv[0]);
        return -1;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  // 检查两种模式参数
  if (model_path && input_image) {
    // 图像模式
    printf("Running in image processing mode:\n");
    printf("  Model path: %s\n  Input image: %s\n", model_path, input_image);
  } else if (bin_data && sample_rate && seconds) {
    // 声音模式
    printf("Running in sound processing mode:\n");
    printf(
        "  Sound model: %s\n  Bin data: %s\n  Sample rate: %s\n  Duration: %s "
        "sec\n",
        model_path, bin_data, sample_rate, seconds);
  } else if (bin_data && data_count) {
    // 特征点模式
    printf("Running in keypoint processing mode:\n");
    printf("  Keypoint model: %s\n  Bin data: %s\n  Data count: %s\n\n",
           model_path, bin_data, data_count);
  } else {
    fprintf(stderr, "Error: Invalid combination of parameters\n");
    print_usage(argv[0]);
    return -1;
  }

  int ret = 0;

  TDLModel model_id;
  if (get_model_info(model_path, &model_id) == -1) {
    printf("unsupported model: %s\n", model_path);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, NULL);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  // The default threshold is 0.5
  ret = TDL_SetModelThreshold(tdl_handle, model_id, 0.5);
  if (ret != 0) {
    printf("TDL_SetModelThreshold failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image;

  if (input_image) {
    image = TDL_ReadImage(input_image);
    if (image == NULL) {
      printf("read image failed with %#x!\n", ret);
      goto exit1;
    }
  } else {
    const char *bin_data_path = bin_data;
    // int size = 0;
    TDLDataTypeE Datatype;
    if (data_count != NULL) {
      // size = atoi(data_count);
      Datatype = TDL_TYPE_FP32;
    } else {
      // int isample_rate = atoi(sample_rate);
      // int iseconds = atoi(seconds);
      Datatype = TDL_TYPE_UINT8;
      // size = isample_rate * AUDIOFORMATSIZE * iseconds;
    }

    image = TDL_ReadBin(bin_data_path, Datatype);
    if (image == NULL) {
      printf("read audio failed with %#x!\n", ret);
      goto exit1;
    }
  }
  TDLClassInfo obj_info = {0};

  ret = TDL_Classification(tdl_handle, model_id, image, &obj_info);
  if (ret != 0) {
    printf("TDL_Classification failed with %#x!\n", ret);
  } else {
    printf("pred_label: %d, score = %f\n", obj_info.class_id, obj_info.score);
  }

  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
