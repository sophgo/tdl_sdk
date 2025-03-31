#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_name, TDLModel *model_index) {
  int ret = 0;
  if (strcmp(model_name, "FACE_VEHICLE") == 0) {
    *model_index = TDL_MODEL_SEG_PERSON_FACE_VEHICLE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -n FACE_VEHICLE -m <model_path> -i <input_image>\n", prog_name);
  printf("  %s --name FACE_VEHICLE --model_path <path> --input <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -n, --name          Model name (must be FACE_VEHICLE)\n");
  printf("  -m, --model_path    Path to instance segmentation model\n");
  printf("  -i, --input         Path to input image\n");
  printf("  -h, --help          Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_name = NULL;
  char *model_path = NULL;
  char *input_image = NULL;

  struct option long_options[] = {
      {"name",         required_argument, 0, 'n'},
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "n:m:i:h", long_options, NULL)) != -1) {
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

  if (!model_name || !model_path || !input_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  if (strcmp(model_name, "FACE_VEHICLE") != 0) {
      fprintf(stderr, "Error: Model name must be FACE_VEHICLE\n");
      return -1;
  }

  printf("Running with:\n");
  printf("  Model name:    %s\n", model_name);
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);

  int ret = 0;

  TDLModel model_id;
  ret = get_model_info(model_name, &model_id);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path);
  if (ret != 0) {
    printf("open instance seg model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLSegmentation seg_meta = {0};
  ret = TDL_SemanticSegmentation(tdl_handle, model_id, image, &seg_meta);
  if (ret != 0) {
    printf("CVI_TDL_InstanceSegmentation failed with %#x!\n", ret);
  } else {
    printf("height : %d, ", seg_meta.height);
    printf("width : %d, ", seg_meta.width);
    printf("output_height : %d, ", seg_meta.output_height);
    printf("output_width : %d\n", seg_meta.output_width);
    for (int x = 0; x < seg_meta.output_height; x ++) {
        for (int y = 0; y < seg_meta.output_width; y ++) {
            printf("%d ", (int)seg_meta.class_id[x * seg_meta.output_width + y]);
        }
        printf("\n");
    }
  }

  TDL_ReleaseSemanticSegMeta(&seg_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}