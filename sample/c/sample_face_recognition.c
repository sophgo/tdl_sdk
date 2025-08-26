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
  } else if (strstr(model_path, "keypoint_face_v2_64_64") != NULL) {
    *model_index = TDL_MODEL_KEYPOINT_FACE_V2;
  } else if (strstr(model_path, "scrfd_det_face") != NULL) {
    *model_index = TDL_MODEL_SCRFD_DET_FACE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m "
      "<face_detect_model_path>,<face_landmark_model_path>,<face_feature_model_"
      "path> -i <input_image>,<input_image> -c config_path\n",
      prog_name);
  printf(
      "  %s  --model_path "
      "<face_detect_model_path>,<face_landmark_model_path>,<face_feature_model_"
      "path> --input <image>,<image> --config "
      "config_path\n\n",
      prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path      Path to detect, landmark and feature model\n");
  printf("  -i, --input           Path to first input image\n");
  printf("  -c, --config          Path to first config\n");
  printf("  -h, --help            Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path1 = NULL;
  char *model_path2 = NULL;
  char *model_path3 = NULL;
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

  if (!input_image || !config || !model_path) {
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

  char *comm = strchr(model_path, ',');
  if (!comm || comm == model_path || !*(comm + 1)) {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path1,model_path2,model_path3\n");
    return -1;
  }

  const char *first_comma = strchr(model_path, ',');
  if (!first_comma || first_comma == model_path || first_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path1,model_path2,model_path3'\n");
    return -1;
  }
  const char *second_comma = strchr(first_comma + 1, ',');
  if (!second_comma || second_comma == first_comma + 1 ||
      second_comma[1] == '\0') {
    fprintf(stderr,
            "Error: Models must be in format "
            "'model_path1,model_path2,model_path3'\n");
    return -1;
  }

  if (strchr(second_comma + 1, ',')) {
    fprintf(stderr, "Error: Exactly three model paths are required\n");
    return -1;
  }

  char *comm1 = (char *)first_comma;
  char *comm2 = (char *)second_comma;

  model_path1 = model_path;
  *comm1 = '\0';
  model_path2 = comm1 + 1;
  *comm2 = '\0';
  model_path3 = comm2 + 1;

  printf("Running with:\n");
  printf("  Model path1:     %s\n", model_path1);
  printf("  Model path2:     %s\n", model_path2);
  printf("  Model path3:     %s\n", model_path3);
  printf("  Input image 1:   %s\n", input_image1);
  printf("  Input image 2:   %s\n", input_image2);
  printf("  Config:          %s\n", config);

  int ret = 0;

  TDLModel model_id_detect;
  ret = get_model_info(model_path1, &model_id_detect);
  if (ret != 0) {
    printf("None detect model name to support\n");
    return -1;
  }

  TDLModel model_id_landmark;
  ret = get_model_info(model_path2, &model_id_landmark);
  if (ret != 0) {
    printf("None landkark model name to support\n");
    return -1;
  }

  TDLModel model_id_feature;
  ret = get_model_info(model_path3, &model_id_feature);
  if (ret != 0) {
    printf("None feature model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id_detect, model_path1, config);
  if (ret != 0) {
    printf("open detect model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_landmark, model_path2, config);
  if (ret != 0) {
    printf("open landkark model failed with %#x!\n", ret);
    goto exit1;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_feature, model_path3, config);
  if (ret != 0) {
    printf("open feature model failed with %#x!\n", ret);
    goto exit2;
  }

  TDLImage image1 = TDL_ReadImage(input_image1);
  if (image1 == NULL) {
    printf("read image1 failed with %#x!\n", ret);
    goto exit3;
  }

  TDLImage image2 = TDL_ReadImage(input_image2);
  if (image2 == NULL) {
    printf("read image2 failed with %#x!\n", ret);
    goto exit3;
  }

  TDLFace face_meta1 = {0}, face_meta2 = {0};
  TDLImage crop_image1, crop_image2;

  ret = TDL_FaceDetection(tdl_handle, model_id_detect, image1, &face_meta1);
  if (ret != 0) {
    printf("TDL_FaceDetection 1 failed with %#x!\n", ret);
    goto exit5;
  }

  ret = TDL_FaceDetection(tdl_handle, model_id_detect, image2, &face_meta2);
  if (ret != 0) {
    printf("TDL_FaceDetection 2 failed with %#x!\n", ret);
    goto exit5;
  }

  ret = TDL_FaceLandmark(tdl_handle, model_id_landmark, image1, &crop_image1,
                         &face_meta1);
  if (ret != 0 || crop_image1 == NULL) {
    printf("TDL_FaceLandmark 1 failed with %#x!\n", ret);
    goto exit5;
  }

  ret = TDL_FaceLandmark(tdl_handle, model_id_landmark, image2, &crop_image2,
                         &face_meta2);
  if (ret != 0 || crop_image2 == NULL) {
    printf("TDL_FaceLandmark 2 failed with %#x!\n", ret);
    goto exit5;
  }

  TDLFeature obj_meta1 = {0}, obj_meta2 = {0};
  ret = TDL_FeatureExtraction(tdl_handle, model_id_feature, crop_image1,
                              &obj_meta1);
  if (ret != 0) {
    printf("TDL_FeatureExtraction 1 failed with %#x!\n", ret);
    goto exit5;
  }

  ret = TDL_FeatureExtraction(tdl_handle, model_id_feature, crop_image2,
                              &obj_meta2);
  if (ret != 0) {
    printf("TDL_FeatureExtraction 2 failed with %#x!\n", ret);
    goto exit5;
  }

  float similarity = 0.0;
  ret = TDL_CaculateSimilarity(obj_meta1, obj_meta2, &similarity);
  printf("similarity is %f\n", similarity);

  if (crop_image1) {
    TDL_DestroyImage(crop_image1);
  }

  if (crop_image2) {
    TDL_DestroyImage(crop_image2);
  }

exit5:
  TDL_ReleaseFeatureMeta(&obj_meta1);
  TDL_ReleaseFeatureMeta(&obj_meta2);
  TDL_ReleaseFaceMeta(&face_meta1);
  TDL_ReleaseFaceMeta(&face_meta2);
  TDL_DestroyImage(image2);

exit4:
  TDL_DestroyImage(image1);

exit3:
  TDL_CloseModel(tdl_handle, model_id_feature);

exit2:
  TDL_CloseModel(tdl_handle, model_id_landmark);

exit1:
  TDL_CloseModel(tdl_handle, model_id_detect);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
