#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "recognition_face_r34") != NULL) {
    *model_index = TDL_MODEL_FEATURE_BMFACE_R34;
  } else if (strstr(model_path, "feature_cviface_112_112_INT8") != NULL) {
    *model_index = TDL_MODEL_FEATURE_CVIFACE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <input_image>,<input_image> -c config_path\n",
         prog_name);
  printf(
      "  %s  --model_path <path> --input <image>,<image> --config "
      "config_path\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --model_path      Path to feature model\n");
  printf("  -i, --input           Path to first input image\n");
  printf("  -c, --config          Path to first config\n");
  printf("  -h, --help            Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image1 = NULL;
  char *input_image2 = NULL;
  char *input_image = NULL;
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
        print_usage(argv[0]);
        return -1;
      default:
        print_usage(argv[0]);
        return -1;
    }
  }

  if (!input_image || !config) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(input_image, ',');
  if (!comma || comma == input_image || !*(comma + 1)) {
    fprintf(stderr, "Error: Models must be in format 'image1,image2'\n");
    return -1;
  }
  input_image1 = input_image;
  *comma = '\0';
  input_image2 = comma + 1;

  if (!model_path || !input_image1 || !input_image2) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:      %s\n", model_path);
  printf("  Input image 1:   %s\n", input_image1);
  printf("  Input image 2:   %s\n", input_image2);
  printf("  Config:          %s\n", config);

  int ret = 0;

  TDLModel model_id;
  ret = get_model_info(model_path, &model_id);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, config, 0);
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
