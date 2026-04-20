#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "recognition_license_plate") != NULL) {
    *model_index = TDL_MODEL_RECOGNITION_LICENSE_PLATE;
  } else if (strstr(model_path, "keypoint_license_plate") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strstr(model_path, "yolov8n_det_license_plate") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m "
      "<detect_model_path>,<keypoint_model_path>,<recognition_model_path> -i "
      "<input_image>\n",
      prog_name);
  printf(
      "  %s --model_path "
      "<detect_model_path>,<keypoint_model_path>,<recognition_model_path> "
      "--input <image>\n\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Path to detect, keypoint and recognition model\n");
  printf("  -i, --input       Path to input image\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path_detect = NULL;
  char *model_path_keypoint = NULL;
  char *model_path_recognition = NULL;
  char *model_path = NULL;
  char *input_image = NULL;
  char *output_image = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input", required_argument, 0, 'i'},
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
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comm = strchr(model_path, ',');
  if (!comm || comm == model_path || !*(comm + 1)) {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition\n");
    return -1;
  }

  const char *first_comma = strchr(model_path, ',');
  if (!first_comma || first_comma == model_path || first_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }
  const char *second_comma = strchr(first_comma + 1, ',');
  if (!second_comma || second_comma == first_comma + 1 ||
      second_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path_detect,model_path_keypoint,model_path_recognition'\n");
    return -1;
  }

  if (strchr(second_comma + 1, ',')) {
    fprintf(stderr, "Error: Exactly three model paths are required\n");
    return -1;
  }

  char *comm1 = (char *)first_comma;
  char *comm2 = (char *)second_comma;

  model_path_detect = model_path;
  *comm1 = '\0';
  model_path_keypoint = comm1 + 1;
  *comm2 = '\0';
  model_path_recognition = comm2 + 1;

  printf("Running with:\n");
  printf("  Model path_detect:      %s\n", model_path_detect);
  printf("  Model path_keypoint:    %s\n", model_path_keypoint);
  printf("  Model path_recognition: %s\n", model_path_recognition);
  printf("  Input image:            %s\n", input_image);
  printf("  Output image:           %s\n", output_image);

  int ret = 0;

  TDLModel model_id_detect;
  ret = get_model_info(model_path_detect, &model_id_detect);
  if (ret != 0) {
    printf("None detect model name to support\n");
    return -1;
  }

  TDLModel model_id_keypoint;
  ret = get_model_info(model_path_keypoint, &model_id_keypoint);
  if (ret != 0) {
    printf("None keypoint model name to support\n");
    return -1;
  }

  TDLModel model_id_recognition;
  ret = get_model_info(model_path_recognition, &model_id_recognition);
  if (ret != 0) {
    printf("None recognition model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  ret = TDL_OpenModel(tdl_handle, model_id_detect, model_path_detect, NULL, 0);
  if (ret != 0) {
    printf("open detect model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_keypoint, model_path_keypoint, NULL,
                      0);
  if (ret != 0) {
    printf("open keypoint model failed with %#x!\n", ret);
    goto exit1;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_recognition, model_path_recognition,
                      NULL, 0);
  if (ret != 0) {
    printf("open recognition model failed with %#x!\n", ret);
    goto exit2;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit3;
  }

  TDLObject obj_meta = {0};
  ret = TDL_Detection(tdl_handle, model_id_detect, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
    goto exit3;
  } else if (obj_meta.size <= 0) {
    printf("None to detection\n");
    goto exit3;
  }

  TDLImage *crop_image = (TDLImage *)malloc(sizeof(TDLImage) * obj_meta.size);
  ret = TDL_DetectionKeypoint(tdl_handle, model_id_keypoint, image, &obj_meta,
                              crop_image);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    for (int32_t i = 0; i < obj_meta.size; i++) {
      TDLText ocr_meta = {0};
      ret = TDL_CharacterRecognition(tdl_handle, model_id_recognition,
                                     crop_image[i], &ocr_meta);
      if (ret != 0) {
        printf("TDL_CharacterRecognition failed with %#x!\n", ret);
      } else {
        if (ocr_meta.size <= 0) {
          printf("None to detection\n");
        } else {
          printf("id = %d, txt info: %s\n", i, ocr_meta.text_info);
        }
      }
      TDL_ReleaseCharacterMeta(&ocr_meta);
    }
  }

  TDL_ReleaseObjectMeta(&obj_meta);
  for (int32_t i = 0; i < obj_meta.size; i++) {
    TDL_DestroyImage(crop_image[i]);
  }
  free(crop_image);
  crop_image = NULL;

exit3:
  TDL_CloseModel(tdl_handle, model_id_recognition);

exit2:
  TDL_CloseModel(tdl_handle, model_id_keypoint);

exit1:
  TDL_CloseModel(tdl_handle, model_id_detect);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
