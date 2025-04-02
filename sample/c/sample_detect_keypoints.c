#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index_d,  TDLModel *model_index_k) {
  int ret = 0;
  if (strstr(model_path, "keypoint_hand") != NULL) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_HAND;
    *model_index_k = TDL_MODEL_KEYPOINT_HAND;
  } else if (strstr(model_path, "keypoint_license_plate") != NULL) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
    *model_index_k = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strstr(model_path, "keypoint_simcc_person17") != NULL) {
    *model_index_d = TDL_MODEL_MBV2_DET_PERSON;
    *model_index_k = TDL_MODEL_KEYPOINT_SIMICC;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <detect_model>,<kp_model> -i <input_image>\n\n", prog_name);
  printf("  %s --model_path <detect_path>,<kp_path> --input <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Comma-separated model paths\n"
         "  <yolov8n_det_hand_xxx,keypoint_hand_xxx>\n"
         "  <license_plate_detection_yolov8n_xxx,keypoint_license_plate_xxx>\n"
         "  <mbv2_det_person_xxx,keypoint_simcc_person17_xxx>\n");
  printf("  -i, --input       Path to input image\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *detect_model = NULL;
  char *kp_model = NULL;
  char *input_image = NULL;
  char *models = NULL;

  struct option long_options[] = {
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"name",         required_argument, 0, 'n'},
      {"help",         no_argument,       0, 'h'},
      {0, 0, 0, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:h", long_options, NULL)) != -1) {
      switch (opt) {
          case 'm': 
              models = optarg;
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

  if (!models || !input_image) {
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
  if (!detect_model || !kp_model || !input_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Detection model: %s\n", detect_model);
  printf("  Keypoint model:  %s\n", kp_model);
  printf("  Input image:     %s\n", input_image);
  
  int ret = 0;

  TDLModel model_id_d, model_id_k;
  ret = get_model_info(kp_model, &model_id_d, &model_id_k);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id_d, detect_model);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_k, kp_model);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit2;
  }

  TDLObject obj_meta = {0};

  ret = TDL_Detection(tdl_handle, model_id_d, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
    goto exit3;
  } else if (obj_meta.size <= 0){
      printf("None to detection\n");
      goto exit3;
  }

  ret = TDL_DetectionKeypoint(tdl_handle, model_id_k, image, &obj_meta);
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
  TDL_CloseModel(tdl_handle, model_id_k);

exit1:
  TDL_CloseModel(tdl_handle, model_id_d);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
