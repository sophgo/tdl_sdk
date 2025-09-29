#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "clip_image") != NULL) {
    *model_index = TDL_MODEL_FEATURE_CLIP_IMG;
  } else if (strstr(model_path, "clip_text") != NULL) {
    *model_index = TDL_MODEL_FEATURE_CLIP_TEXT;
  } else if (strstr(model_path, "mobileclip2_B_img") != NULL) {
    *model_index = TDL_MODEL_FEATURE_MOBILECLIP2_IMG;
  } else if (strstr(model_path, "mobileclip2_B_text") != NULL) {
    *model_index = TDL_MODEL_FEATURE_MOBILECLIP2_TEXT;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m <image_model_path> -n <text_model_path> -i <input_image> -t "
      "<txt_dir>\n",
      prog_name);
  printf(
      "  %s --image_model_path <path> --text_model_path <path> --input <image> "
      "--txt_dir <path>\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --image_model_path   Path to clip image model\n");
  printf("  -n, --text_model_path   Path to clip text model\n");
  printf("  -i, --input        Path to input image\n");
  printf("  -t, --txt_dir      Path to txt directory\n");
  printf("  -h, --help         Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *image_model_path = NULL;
  char *text_model_path = NULL;
  char *input_image = NULL;
  char *txt_dir = NULL;

  struct option long_options[] = {
      {"image_model_path", required_argument, 0, 'm'},
      {"text_model_path", required_argument, 0, 'n'},
      {"input", required_argument, 0, 'i'},
      {"txt_dir", required_argument, 0, 't'},
      {"help", no_argument, 0, 'h'},
      {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:n:i:t:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'm':
        image_model_path = optarg;
        break;
      case 'n':
        text_model_path = optarg;
        break;
      case 'i':
        input_image = optarg;
        break;
      case 't':
        txt_dir = optarg;
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

  if (!image_model_path || !input_image || !text_model_path || !txt_dir) {
    fprintf(stderr,
            "Error: image model path, text model path, input image and txt "
            "directory are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Image model path:    %s\n", image_model_path);
  printf("  Text model path:    %s\n", text_model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Txt directory:   %s\n", txt_dir);

  int ret = 0;

  TDLModel model_id1;
  if (get_model_info(image_model_path, &model_id1) == -1) {
    printf("unsupported model: %s\n", image_model_path);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id1, image_model_path, NULL);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
  }

  TDLModel model_id2;
  if (get_model_info(text_model_path, &model_id2) == -1) {
    printf("unsupported model: %s\n", text_model_path);
    return -1;
  }
  ret = TDL_OpenModel(tdl_handle, model_id2, text_model_path, NULL);

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
  }

  TDLFeature obj_meta1 = {0};
  ret = TDL_FeatureExtraction(tdl_handle, model_id1, image, &obj_meta1);

  float *fptr = (float *)(obj_meta1.ptr);

  float *feature_out = NULL;
  int numSentences;
  int embedding_num;
  ret = TDL_ClipText(tdl_handle, model_id2, txt_dir, &feature_out,
                     &numSentences, &embedding_num);
  int image_rows = 1;
  float *result = NULL;
  ret = TDL_ClipPostprocess(feature_out, numSentences, fptr, image_rows,
                            embedding_num, &result);
  for (uint32_t i = 0; i < numSentences; ++i) {
    printf("%f\n", result[i]);
  }
  TDL_DestroyImage(image);

  TDL_CloseModel(tdl_handle, model_id1);

  TDL_DestroyHandle(tdl_handle);

  return ret;
}
