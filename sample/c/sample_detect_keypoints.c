#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "tdl_sdk.h"
#include "tdl_utils.h"

int get_model_info(char *model_path, TDLModel *model_index_d,
                   TDLModel *model_index_k) {
  int ret = 0;
  if (strstr(model_path, "keypoint_hand") != NULL) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_HAND;
    *model_index_k = TDL_MODEL_KEYPOINT_HAND;
  } else if (strstr(model_path, "keypoint_license_plate") != NULL) {
    *model_index_d = TDL_MODEL_YOLOV8N_DET_LICENSE_PLATE;
    *model_index_k = TDL_MODEL_KEYPOINT_LICENSE_PLATE;
  } else if (strstr(model_path, "keypoint_simcc_person17") != NULL) {
    *model_index_d = TDL_MODEL_MBV2_DET_PERSON;
    *model_index_k = TDL_MODEL_KEYPOINT_SIMCC_PERSON17;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m <detect_model>,<kp_model> -i <input_image> "
      "-o <output_image_crop>,<output_image>\n\n",
      prog_name);
  printf("  %s --model_path <detect_path>,<kp_path> --input <image>\n\n",
         prog_name);
  printf("Options:\n");
  printf(
      "  -m, --model_path  Comma-separated model paths\n"
      "  <yolov8n_det_hand_xxx,keypoint_hand_xxx>\n"
      "  <yolov8n_det_license_plate_xxx,keypoint_license_plate_xxx>\n"
      "  <mbv2_det_person_xxx,keypoint_simcc_person17_xxx>\n");
  printf("  -i, --input       Path to input image\n");
  printf("  -h, --help        Show this help message\n");
  printf("  -o, --output         Path to output image\n");
}

int main(int argc, char *argv[]) {
  char *detect_model = NULL;
  char *kp_model = NULL;
  char *input_image = NULL;
  char *models = NULL;
  char *output_image1 = NULL;
  char *output_image2 = NULL;
  char *output_image = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input", required_argument, 0, 'i'},
                                  {"name", required_argument, 0, 'n'},
                                  {"output", required_argument, 0, 'o'},
                                  {"help", no_argument, 0, 'h'},
                                  {0, 0, 0, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:o:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        models = optarg;
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

  if (!models || !input_image) {
    fprintf(stderr, "Error: All arguments are required\n");
    print_usage(argv[0]);
    return -1;
  }

  char *comma = strchr(models, ',');
  if (!comma || comma == models || !*(comma + 1)) {
    fprintf(stderr, "Error: Models must be in format 'detect,attr'\n");
    return -1;
  }
  detect_model = models;
  *comma = '\0';
  kp_model = comma + 1;

  if (output_image != NULL) {
    char *comma_out = strchr(output_image, ',');
    if (!comma_out || comma_out == input_image || !*(comma_out + 1)) {
      fprintf(stderr, "Error: Models must be in format 'image1,image2'\n");
      return -1;
    }
    output_image1 = output_image;
    *comma_out = '\0';
    output_image2 = comma_out + 1;
  }

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
  printf("  output image1:     %s\n", output_image1);
  printf("  output image2:     %s\n", output_image2);

  int ret = 0;

  TDLModel model_id_d, model_id_k;
  ret = get_model_info(kp_model, &model_id_d, &model_id_k);
  if (ret != 0) {
    printf("None model name to support\n");
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id_d, detect_model, NULL);
  if (ret != 0) {
    printf("open hand keypoint model failed with %#x!\n", ret);
    goto exit0;
  }

  ret = TDL_OpenModel(tdl_handle, model_id_k, kp_model, NULL);
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
  } else if (obj_meta.size <= 0) {
    printf("None to detection\n");
    goto exit3;
  }

  char basename[128];
  if (output_image2 != NULL) {
    char *dot = strrchr(output_image2, '.');
    if (dot != NULL) {
      int baselen = dot - output_image2;
      strncpy(basename, output_image2, baselen);
      basename[baselen] = '\0';
    } else {
      strcpy(basename, output_image2);
    }
  }

  ret = TDL_DetectionKeypoint(tdl_handle, model_id_k, image, &obj_meta, NULL);
  if (ret != 0) {
    printf("TDL_KeypointDetection failed with %#x!\n", ret);
  } else {
    box_t boxes[obj_meta.size];
    for (int i = 0; i < obj_meta.size; i++) {
      boxes[i].x1 = obj_meta.info[i].box.x1;
      boxes[i].y1 = obj_meta.info[i].box.y1;
      boxes[i].x2 = obj_meta.info[i].box.x2;
      boxes[i].y2 = obj_meta.info[i].box.y2;
      point_t point[obj_meta.info[i].landmark_size];
      for (int j = 0; j < obj_meta.info[0].landmark_size; j++) {
        printf("obj_meta id: %d, ", i);
        printf("[x, y]: %f, %f\n", obj_meta.info[i].landmark_properity[j].x,
               obj_meta.info[i].landmark_properity[j].y);
        if (strstr(kp_model, "keypoint_hand") != NULL) {
          point[j].x =
              obj_meta.info[i].landmark_properity[j].x * obj_meta.width;
          point[j].y =
              obj_meta.info[i].landmark_properity[j].y * obj_meta.height;
        } else {
          point[j].x = obj_meta.info[i].landmark_properity[j].x;
          point[j].y = obj_meta.info[i].landmark_properity[j].y;
        }
      }
      if (output_image2 != NULL) {
        if (i == 0) {
          TDL_VisualizeRectangle(boxes, obj_meta.size, input_image,
                                 output_image1);
        } else {
          TDL_VisualizeRectangle(boxes, obj_meta.size, output_image1,
                                 output_image1);
        }
        char crop_image[128] = "";
        snprintf(crop_image, sizeof(crop_image), "%s%d.jpg", basename, i);
        int cropX = (int)obj_meta.info[i].box.x1 -
                    (int)round((0.25 * obj_meta.width) / 2);
        int cropY = (int)obj_meta.info[i].box.y1 -
                    (int)round((0.25 * obj_meta.height) / 2);
        TDL_CropImage(cropX, cropY, obj_meta.width, obj_meta.height,
                      output_image1, crop_image);
        TDL_VisualizePoint(point, obj_meta.info[i].landmark_size, crop_image,
                           crop_image);
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
