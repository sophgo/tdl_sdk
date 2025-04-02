#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <detect_model>,<attr_model> -i <input_image>\n", prog_name);
  printf("  %s --model_path <detect_path>,<attr_path> --input <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path     Comma-separated model paths"
         "<scrfd_det_face_xxx,cls_attribute_face_xxx>\n");
  printf("  -i, --input      Path to input image\n");
  printf("  -h, --help       Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *detect_model = NULL;
  char *attr_model = NULL;
  char *input_image = NULL;
  char *models = NULL;

  struct option long_options[] = {
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
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
  attr_model = comma + 1;

  if (!detect_model || !attr_model || !input_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Face detection model: %s\n", detect_model);
  printf("  Face attribute model: %s\n", attr_model);
  printf("  Input image:          %s\n", input_image);

  int ret = 0;

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, TDL_MODEL_SCRFD_DET_FACE, detect_model);
  if (ret != 0) {
    printf("open face detection model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, TDL_MODEL_CLS_ATTRIBUTE_FACE, attr_model);
  if (ret != 0) {
    printf("open face attribute model failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit2;
  }

  TDLFace obj_meta = {0};

  ret = TDL_FaceDetection(tdl_handle, TDL_MODEL_SCRFD_DET_FACE, image, &obj_meta);
  if(ret != 0) {
    printf("TDL_FaceDetection failed with %#x!\n", ret);
    goto exit3;
  }

  ret = TDL_FaceAttribute(tdl_handle, TDL_MODEL_CLS_ATTRIBUTE_FACE, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_FaceAttribute failed with %#x!\n", ret);
  } else {
    printf("gender score:%f,age score:%f,glass score:%f,mask score:%f\n",
        obj_meta.info->gender_score, obj_meta.info->age,
        obj_meta.info->glass_score, obj_meta.info->mask_score);
    printf("Gender:%s\n", obj_meta.info->gender_score > 0.5 ? "Male" : "Female");
    printf("Age:%d\n", (int)round(obj_meta.info->age * 100.0));
    printf("Glass:%s\n", obj_meta.info->glass_score > 0.5 ? "Yes" : "No");
    printf("Mask:%s\n", obj_meta.info->mask_score > 0.5 ? "Yes" : "No");
  }

exit3:
  TDL_ReleaseFaceMeta(&obj_meta);
  TDL_DestroyImage(image);
exit2:
  TDL_CloseModel(tdl_handle, TDL_MODEL_CLS_ATTRIBUTE_FACE);
exit1:
  TDL_CloseModel(tdl_handle, TDL_MODEL_SCRFD_DET_FACE);
exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
