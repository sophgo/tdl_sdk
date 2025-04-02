#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_path, TDLModel *model_index) {
  int ret = 0;
  if (strstr(model_path, "yolov8n_det_person_vehicle") != NULL) {
    *model_index = TDL_MODEL_YOLOV8N_DET_PERSON_VEHICLE;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -i <input_image> -o <output_image>\n", prog_name);
  printf("  %s --model_path <path> --input <image> --output <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Path to cvimodel"
         "<yolov8n_det_person_vehicle>\n");
  printf("  -i, --input       Path to input image\n");
  printf("  -o, --output      Path to output image\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_image = NULL;
  char *output_image = NULL;

  struct option long_options[] = {
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"output",       required_argument, 0, 'o'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:o:h", long_options, NULL)) != -1) {
      switch (opt) {
          case 'm':
              model_path = optarg;
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

  // 检查必需参数
  if (!model_path || !input_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Output image:  %s\n", output_image);

  int ret = 0;

  TDLModel model_id;
  if (get_model_info(model_path, &model_id) == -1) {
    printf("unsupported model: %s\n", model_path);
    return -1;
  }

  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLObject obj_meta = {0};
  ret = TDL_Detection(tdl_handle, model_id, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Detection failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
      printf("None to detection\n");
    } else {
      box_t boxes[obj_meta.size];
      for (int i = 0; i < obj_meta.size; i++) {
        printf("obj_meta_index : %d, ", i);
        printf("class_id : %d, ", obj_meta.info[i].class_id);
        printf("score : %f, ", obj_meta.info[i].score);
        printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                                          obj_meta.info[i].box.x2,
                                          obj_meta.info[i].box.y1,
                                          obj_meta.info[i].box.y2);
        boxes[i].x1 = obj_meta.info[i].box.x1;
        boxes[i].y1 = obj_meta.info[i].box.y1;
        boxes[i].x2 = obj_meta.info[i].box.x2;
        boxes[i].y2 = obj_meta.info[i].box.y2;
      }
      if (output_image != NULL) {
        TDL_VisualizeRectangle(boxes, obj_meta.size, input_image, output_image);
      }
    }
  }

  TDL_ReleaseObjectMeta(&obj_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
