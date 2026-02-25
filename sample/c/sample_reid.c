#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

static int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;

  if (strstr(model_path, "osnet_cv181x_int8_sym") != NULL) {
    *model_index = TDL_MODEL_FEATURE_REID;
  } else {
    ret = -1;
  }

  return ret;
}

static void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <image1>,<image2> -c <config_path>\n",
         prog_name);
  printf(
      "  %s --model_path <path> --input <image1>,<image2> --config "
      "config_path\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --model_path      Path to ReID feature model\n");
  printf("  -i, --input           Two images, format: image1,image2\n");
  printf("  -c, --config          Model config json path\n");
  printf("  -h, --help            Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image = NULL;
  char *input_image1 = NULL;
  char *input_image2 = NULL;
  char *input_copy = NULL;
  char *config = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input", required_argument, 0, 'i'},
                                  {"config", required_argument, 0, 'c'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:c:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'i':
        input_image = optarg;
        break;
      case 'c':
        config = optarg;
        break;
      case 'h':
        print_usage(argv[0]);
        return 0;
      case '?':
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (!model_path || !input_image || !config) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(input_image, ',');
  if (!comma || comma == input_image || !*(comma + 1)) {
    fprintf(stderr, "Error: Input must be in format 'image1,image2'\n");
    return -1;
  }

  input_copy = strdup(input_image);
  if (!input_copy) {
    fprintf(stderr, "Error: failed to allocate memory for input image paths\n");
    return -1;
  }

  comma = strchr(input_copy, ',');
  *comma = '\0';
  input_image1 = input_copy;
  input_image2 = comma + 1;

  printf("Running ReID C sample with:\n");
  printf("  Model path:      %s\n", model_path);
  printf("  Input image 1:   %s\n", input_image1);
  printf("  Input image 2:   %s\n", input_image2);
  printf("  Config:          %s\n", config);

  int ret = 0;

  TDLModel model_id;
  ret = get_model_info(model_path, &model_id);
  if (ret != 0) {
    printf("Unsupported model path: %s\n", model_path);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    printf("TDL_CreateHandle failed\n");
    return -1;
  }

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, config, 0);
  if (ret != 0) {
    printf("TDL_OpenModel failed with %#x\n", ret);
    goto exit0;
  }

  TDLImage image1 = TDL_ReadImage(input_image1);
  if (image1 == NULL) {
    printf("TDL_ReadImage image1 failed\n");
    ret = -1;
    goto exit1;
  }

  TDLImage image2 = TDL_ReadImage(input_image2);
  if (image2 == NULL) {
    printf("TDL_ReadImage image2 failed\n");
    ret = -1;
    goto exit2;
  }

  TDLFeature feature1 = {0};
  TDLFeature feature2 = {0};

  ret = TDL_FeatureExtraction(tdl_handle, model_id, image1, &feature1);
  if (ret != 0) {
    printf("TDL_FeatureExtraction image1 failed with %#x\n", ret);
    goto exit3;
  }

  ret = TDL_FeatureExtraction(tdl_handle, model_id, image2, &feature2);
  if (ret != 0) {
    printf("TDL_FeatureExtraction image2 failed with %#x\n", ret);
    goto exit3;
  }

  float similarity = 0.0f;
  ret = TDL_CaculateSimilarity(feature1, feature2, &similarity);
  if (ret != 0) {
    printf("TDL_CaculateSimilarity failed with %#x\n", ret);
  } else {
    printf("ReID similarity: %f\n", similarity);
  }

exit3:
  TDL_ReleaseFeatureMeta(&feature1);
  TDL_ReleaseFeatureMeta(&feature2);
  TDL_DestroyImage(image2);

exit2:
  TDL_DestroyImage(image1);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  if (input_copy) {
    free(input_copy);
  }
  return ret;
}
