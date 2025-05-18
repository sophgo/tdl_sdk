#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "keypoint_hand") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_HAND;
  } else if (strstr(model_path, "keypoint_license_plate") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strstr(model_path, "keypoint_simcc_person17") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_SIMCC_PERSON17;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <input_image> -o <output_image>\n",
         prog_name);
  printf("  %s --model_path <path> --input <image> --output <image>\n\n",
         prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path     Path to keypoint model\n"
      "  <keypoint_hand_xxx>\n"
      "  <keypoint_license_plate>\n"
      "  <keypoint_simcc_person17>\n");
  printf("  -i, --input          Path to input image\n");
  printf("  -o, --output         Path to output image\n");
  printf("  -h, --help           Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image = NULL;
  char *output_image = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input", required_argument, 0, 'i'},
                                  {"output", required_argument, 0, 'o'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:o:h", long_options, NULL)) != -1) {
    switch (opt) {
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
  if (!model_path || !input_image) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Output image:  %s\n", output_image);

  int ret = 0;

  TDLModel model_id;
  ret = get_model_info(model_path, &model_id);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, NULL);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLKeypoint obj_meta = {0};
  ret = TDL_Keypoint(tdl_handle, model_id, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
      printf("None to detection\n");
    } else {
      point_t point[obj_meta.size];
      for (int i = 0; i < obj_meta.size; i++) {
        printf("obj_meta id : %d, ", i);
        printf("[x, y, score] : %f, %f\n", obj_meta.info[i].x * obj_meta.width,
               obj_meta.info[i].y * obj_meta.height);
        point[i].x = obj_meta.info[i].x * obj_meta.width;
        point[i].y = obj_meta.info[i].y * obj_meta.height;
      }
      if (output_image != NULL) {
        TDL_VisualizePoint(point, obj_meta.size, input_image, output_image);
      }
    }
  }

  TDL_ReleaseKeypointMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
