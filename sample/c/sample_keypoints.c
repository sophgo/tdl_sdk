#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_name, tdl_model_e *model_index) {
  int ret = 0;
  if (strcmp(model_name, "HAND") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_HAND;
  } else if (strcmp(model_name, "LICENSE_PLATE") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strcmp(model_name, "POSE_SIMCC") == 0) {
    *model_index = TDL_MODEL_KEYPOINT_SIMICC;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -n <model> -m <model_path> -i <input_image> -o <output_image>\n", prog_name);
  printf("  %s --name <model> --model_path <path> --input <image> --output <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -n, --name           Keypoint model name (HAND, LICENSE_PLATE, POSE_SIMCC)\n");
  printf("  -m, --model_path     Path to keypoint model\n");
  printf("  -i, --input          Path to input image\n");
  printf("  -o, --output         Path to output image\n");
  printf("  -h, --help           Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_name = NULL;
  char *model_path = NULL;
  char *input_image = NULL;
  char *output_image = NULL;

  struct option long_options[] = {
      {"name",         required_argument, 0, 'n'},
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"output",       required_argument, 0, 'o'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "n:m:i:o:h", long_options, NULL)) != -1) {
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
          case 'o':
              output_image = optarg;
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

  // 检查必需参数
  if (!model_name || !model_path || !input_image || !output_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Model name:    %s\n", model_name);
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Output image:  %s\n", output_image);

  int ret = 0;

  tdl_model_e enOdModelId;
  ret = get_model_info(model_name, &enOdModelId);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId, model_path);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  tdl_image_t image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_keypoint_t obj_meta = {0};
  ret = TDL_Keypoint(tdl_handle, enOdModelId, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
      printf("None to detection\n");
    } else {
      point_t point[obj_meta.size];
      for (int i = 0; i < obj_meta.size; i++) {
        printf("obj_meta id : %d, ", i);
        printf("[x, y, score] : %f, %f, %f\n", obj_meta.info[i].x * obj_meta.width,
                                               obj_meta.info[i].y * obj_meta.height,
                                               obj_meta.info[i].score);
        point[i].x = obj_meta.info[i].x * obj_meta.width;
        point[i].y = obj_meta.info[i].y * obj_meta.height;
      }
      TDL_VisualizePoint(point, obj_meta.size, input_image, output_image);
    }
  }

  TDL_ReleaseKeypointMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
