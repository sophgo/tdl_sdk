#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

#define AUDIOFORMATSIZE 2

int get_model_info(char *model_name, TDLModel *model_index) {
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

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  Image processing mode:\n");
  printf("    %s -n <name> -m <model_path> -i <input_image>\n", prog_name);
  printf("    %s --name <name> --model_path <path> --input <image>\n", prog_name);
  printf("  Sound processing mode:\n");
  printf("    %s -n <name> -m <model_path> -b <input_bin> -r <rate> -t <time>\n", prog_name);
  printf("    %s --name <name> --model_path <path> --bin_data <input_bin> --sample-rate <rate> --seconds <time>\n", prog_name);
  printf("\nOptions:\n");
  printf("  -n, --name           Model name (CLS_SOUND_BABAY_CRY, CLS_SOUND_COMMAND, CLS_RGBLIVENESS)\n");
  printf("  -m, --model_path     Path to cvimodel (image mode)\n");
  printf("  -i, --input          Path to input image (image mode)\n");
  printf("  -b, --bin_data       Path to input data (sound mode)\n");
  printf("  -r, --sample-rate    Sample rate in Hz (sound mode)\n");
  printf("  -t, --seconds        Duration in seconds (sound mode)\n");
  printf("  -h, --help           Display this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_name = NULL;
  char *model_path = NULL;
  char *input_image = NULL;
  char *bin_data = NULL;
  char *sample_rate = NULL;
  char *seconds = NULL;

  struct option long_options[] = {
      {"name",          required_argument, 0, 'n'},
      {"model_path",    required_argument, 0, 'm'},
      {"input",         required_argument, 0, 'i'},
      {"bin_data",      required_argument, 0, 'b'},
      {"sample-rate",   required_argument, 0, 'r'},
      {"seconds",       required_argument, 0, 't'},
      {"help",          no_argument,       0, 'h'},
      {0, 0, 0, 0}
  };

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "n:m:i:b:r:t:h", long_options, &option_index)) != -1) {
      switch (opt) {
          case 'n':
              model_name = optarg;
              break;
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
          case 't':
              seconds = optarg;
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

  // 检查必需参数
  if (!model_name) {
      fprintf(stderr, "Error: --model-name is required\n");
      print_usage(argv[0]);
      return -1;
  }

  // 检查两种模式参数
  if (model_path && input_image) {
      // 图像模式
      printf("Running in image processing mode:\n");
      printf("  Model: %s\n  Model path: %s\n  Input image: %s\n", 
             model_name, model_path, input_image);
  } else if (model_name && bin_data && sample_rate && seconds) {
      // 声音模式
      printf("Running in sound processing mode:\n");
      printf("  Model: %s\n  Sound model: %s\n  Bin data: %s\n  Sample rate: %s\n  Duration: %s sec\n",
             model_name, model_name, bin_data, sample_rate, seconds);
  } else {
      fprintf(stderr, "Error: Invalid combination of parameters\n");
      print_usage(argv[0]);
      return -1;
  }

  int ret = 0;

  TDLModel model_id;
  if (get_model_info(model_name, &model_id) == -1) {
    printf("unsupported model: %s\n", model_name);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
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
    int isample_rate = atoi(sample_rate);
    int iseconds = atoi(seconds);
    int size = isample_rate * AUDIOFORMATSIZE * iseconds;

    image = TDL_ReadAudio(bin_data_path, size);
    if (image == NULL) {
      printf("read audio failed with %#x!\n", ret);
      goto exit1;
    }

  }
  TDLClassInfo obj_info = {0};

  ret = TDL_Classfification(tdl_handle, model_id, image, &obj_info);

  if (ret != 0) {
    printf("TDL_Classfification failed with %#x!\n", ret);
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
