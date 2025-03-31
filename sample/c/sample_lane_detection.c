#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <input_image>\n", prog_name);
  printf("  %s --model_path <path> --input <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path    Path to cvimodel\n");
  printf("  -i, --input         Path to input image\n");
  printf("  -h, --help          Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image = NULL;

  struct option long_options[] = {
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:h", long_options, NULL)) != -1) {
      switch (opt) {
          case 'm':
              model_path = optarg;
              break;
          case 'i':
              input_image = optarg;
              break;
          case 'h':
              print_usage(argv[0]);
              return 0;
          case '?':
              print_usage(argv[0]);
              return -1;
          default:
              print_usage(argv[0]);
              return -1;
      }
  }

  if (!model_path || !input_image) {
      fprintf(stderr, "Error: Both model path and input image are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);

  int ret = 0;

  TDLModel model_id = TDL_MODEL_LSTR_DET_LANE;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path);
  if (ret != 0) {
    printf("open lane model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLLane obj_meta = {0};
  ret = TDL_LaneDetection(tdl_handle, model_id, image, &obj_meta);
  if (ret != 0) {
    printf("CVI_TDL_LaneDetection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
        printf("None to detection\n");
    } else {
       for (int i = 0; i < obj_meta.size; i ++) {
            printf("lane %d\n", i);
            for (int j = 0; j < 2; j ++) {
                printf("%d: %f %f\n", j, obj_meta.lane[i].x[j], obj_meta.lane[i].y[j]);
            }
       }
    }
  }

  TDL_ReleaseLaneMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
