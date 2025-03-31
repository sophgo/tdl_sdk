#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_name, TDLModel *model_index) {
  int ret = 0;
  if (strcmp(model_name, "FEATURE_FACE") == 0) {
    *model_index = TDL_MODEL_FEATURE_BMFACER34;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -n FEATURE_FACE -m <model_path> -i <input_image>,<input_image>\n", prog_name);
  printf("  %s --name FEATURE_FACE --model_path <path> --input <image>,<image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -n, --name            Model name (FEATURE_FACE)\n");
  printf("  -m, --model_path      Path to feature model\n");
  printf("  -i, --input           Path to first input image\n");
  printf("  -h, --help            Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_name = NULL;
  char *model_path = NULL;
  char *input_image1 = NULL;
  char *input_image2 = NULL;
  char *input_image = NULL;

  struct option long_options[] = {
      {"name",          required_argument, 0, 'n'},
      {"model_path",    required_argument, 0, 'm'},
      {"input",         required_argument, 0, 'i'},
      {"help",          no_argument,       0, 'h'},
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

  if (!input_image) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(input_image, ',');
  if (!comma || comma == input_image || !*(comma+1)) {
      fprintf(stderr, "Error: Models must be in format 'detect,attr'\n");
      return -1;
  }
  input_image1 = input_image;
  *comma = '\0';  
  input_image2 = comma + 1;

  if (!model_name || !model_path || !input_image1 || !input_image2) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Model name:      %s\n", model_name);
  printf("  Model path:      %s\n", model_path);
  printf("  Input image 1:   %s\n", input_image1);
  printf("  Input image 2:   %s\n", input_image2);

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
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image1 = TDL_ReadImage(input_image1);
  if (image1 == NULL) {
    printf("read image1 failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image2 = TDL_ReadImage(input_image2);
  if (image2 == NULL) {
    printf("read image2 failed with %#x!\n", ret);
    goto exit2;
  }

  TDLFeature obj_meta1 = {0}, obj_meta2 = {0};
  ret = TDL_FeatureExtraction(tdl_handle, model_id, image1, &obj_meta1);
  if (ret != 0) {
    printf("TDL_FeatureExtraction failed with %#x!\n", ret);
    goto exit3;
  }

  ret = TDL_FeatureExtraction(tdl_handle, model_id, image2, &obj_meta2);
  if (ret != 0) {
    printf("TDL_FeatureExtraction failed with %#x!\n", ret);
    goto exit3;
  }

  float similarity = 0.0;
  ret = TDL_CaculateSimilarity(obj_meta1, obj_meta2, &similarity);
  printf("similarity is %f\n", similarity);

exit3:
  TDL_ReleaseFeatureMeta(&obj_meta1);
  TDL_ReleaseFeatureMeta(&obj_meta2);
  TDL_DestroyImage(image2);

exit2:
  TDL_DestroyImage(image1);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
