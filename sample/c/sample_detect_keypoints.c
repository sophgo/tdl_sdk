#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_name, tdl_model_e *model_index_d,  tdl_model_e *model_index_k) {
  int ret = 0;
  if (strcmp(model_name, "HAND") == 0) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_HAND;
    *model_index_k = TDL_MODEL_KEYPOINT_HAND;
  } else if (strcmp(model_name, "LICENSE_PLATE") == 0) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
    *model_index_k = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strcmp(model_name, "POSE_SIMCC") == 0) {
    *model_index_d = TDL_MODEL_MBV2_DET_PERSON;
    *model_index_k = TDL_MODEL_KEYPOINT_SIMICC;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <detect_model>,<kp_model> -i <input_image> -n <name>\n\n", prog_name);
  printf("  %s --model_path <detect_path>,<kp_path> --input <image> --name <name>\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Comma-separated model paths (detection,keypoint)\n");
  printf("  -i, --input       Path to input image\n");
  printf("  -n, --name  Model name (HAND, LICENSE_PLATE, POSE_SIMCC)\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *detect_model = NULL;
  char *kp_model = NULL;
  char *input_image = NULL;
  char *model_name = NULL;
  char *models = NULL;

  struct option long_options[] = {
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"name",         required_argument, 0, 'n'},
      {"help",         no_argument,       0, 'h'},
      {0, 0, 0, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:n:h", long_options, NULL)) != -1) {
      switch (opt) {
          case 'm': 
              models = optarg;
              break;
          case 'i':
              input_image = optarg;
              break;
          case 'n':
              model_name = optarg;
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

  if (!models || !input_image || !model_name) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(models, ',');
  if (!comma || comma == models || !*(comma+1)) {
      fprintf(stderr, "Error: Models must be in format 'detect,attr'\n");
      return -1;
  }
  detect_model = models;
  *comma = '\0';  
  kp_model = comma + 1;

  // Validate required arguments
  if (!detect_model || !kp_model || !input_image || !model_name) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  // Validate model type
  if (strcmp(model_name, "HAND") != 0 && 
      strcmp(model_name, "LICENSE_PLATE") != 0 && 
      strcmp(model_name, "POSE_SIMCC") != 0) {
      fprintf(stderr, "Error: Invalid model type. Must be one of: HAND, LICENSE_PLATE, POSE_SIMCC\n");
      return -1;
  }

  printf("Running with:\n");
  printf("  Detection model: %s\n", detect_model);
  printf("  Keypoint model:  %s\n", kp_model);
  printf("  Input image:     %s\n", input_image);
  printf("  Model type:      %s\n", model_name);
  
  int ret = 0;

  tdl_model_e enOdModelId_d, enOdModelId_k;
  ret = get_model_info(model_name, &enOdModelId_d, &enOdModelId_k);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  tdl_handle_t tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, enOdModelId_d, detect_model);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, enOdModelId_k, kp_model);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit1;
  }

  tdl_image_t image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit2;
  }

  tdl_object_t obj_meta = {0};

  ret = TDL_Detection(tdl_handle, enOdModelId_d, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
    goto exit3;
  } else if (obj_meta.size <= 0){
      printf("None to detection\n");
      goto exit3;
  }

  ret = TDL_DetectionKeypoint(tdl_handle, enOdModelId_k, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    for (int i = 0; i < obj_meta.size; i++) {
      for (int j = 0; j < obj_meta.info[0].landmark_size; j++) {
          printf("obj_meta id: %d, ", i);
          printf("[x, y]: %f, %f\n", obj_meta.info[i].landmark_properity[j].x * obj_meta.width,
                                     obj_meta.info[i].landmark_properity[j].y * obj_meta.height);
      }
    }
  }

exit3:
  TDL_ReleaseObjectMeta(&obj_meta);
  TDL_DestroyImage(image);

exit2:
  TDL_CloseModel(tdl_handle, enOdModelId_k);

exit1:
  TDL_CloseModel(tdl_handle, enOdModelId_d);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
