#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

int get_model_info(char *model_name, TDLModel *model_index) {
  int ret = 0;
  if (strcmp(model_name, "YOLOV8_COCO80") == 0) {
    *model_index = TDL_MODEL_SEG_YOLOV8_COCO80;
  } else {
    ret = -1;
  }
  return ret;
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -n YOLOV8_COCO80 -m <model_path> -i <input_image> -o <output_image>\n", prog_name);
  printf("  %s --name YOLOV8_COCO80 --model_path <path> --input <image> --output <image>\n\n", prog_name);
  printf("Options:\n");
  printf("  -n, --name         Model name (YOLOV8_COCO80)\n");
  printf("  -m, --model_path   Path to instance segmentation model\n");
  printf("  -i, --input        Path to input image\n");
  printf("  -o, --output       Path to output image\n");
  printf("  -h, --help         Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_name = NULL;
  char *model_path = NULL;
  char *input_image = NULL;
  char *output_image = NULL;

  struct option long_options[] = {
      {"name",         required_argument, 0, 'n'},
      {"model_path",   required_argument, 0, 'm'},
      {"input",        required_argument, 0, 'i'},
      {"output",       required_argument, 0, 'o'},
      {"help",         no_argument,       0, 'h'},
      {NULL, 0, NULL, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "n:m:i:o:h", long_options, NULL)) != -1) {
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

  // 验证参数
  if (!model_name || !model_path || !input_image || !output_image) {
      fprintf(stderr, "Error: All arguments are required\n");
      print_usage(argv[0]);
      return -1;
  }
  
  printf("Running with:\n");
  printf("  Model name:    %s\n", model_name);
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Output image:  %s\n", output_image);

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
    printf("open instance seg model failed with %#x!\n", ret);
    goto exit0;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLInstanceSeg inst_seg_meta = {0};
  ret = TDL_InstanceSegmentation(tdl_handle, model_id, image, &inst_seg_meta);
  if (ret != 0) {
    printf("TDL_InstanceSegmentation failed with %#x!\n", ret);
  } else {
    if (inst_seg_meta.size <= 0) {
      printf("None to Segmentation\n");
    } else {
        for (int i = 0; i < inst_seg_meta.size; i++) {
            printf("inst_seg_meta_index : %d, ", i);
            printf("box [x1, y1, x2, y2] : %f %f %f %f, ",
              inst_seg_meta.info[i].obj_info->box.x1,
              inst_seg_meta.info[i].obj_info->box.y1,
              inst_seg_meta.info[i].obj_info->box.x2,
              inst_seg_meta.info[i].obj_info->box.y2);
            printf("class : %d, ", inst_seg_meta.info[i].obj_info->class_id);
            printf("score : %f\n", inst_seg_meta.info[i].obj_info->score);
            printf("points=[");
            point_t point[inst_seg_meta.info[i].mask_point_size];
            for (int j = 0; j < inst_seg_meta.info[i].mask_point_size; j++) {
              printf("(%f,%f),", inst_seg_meta.info[i].mask_point[2 * j],
                inst_seg_meta.info[i].mask_point[2 * j + 1]);
              point[j].x = inst_seg_meta.info[i].mask_point[2 * j];
              point[j].y = inst_seg_meta.info[i].mask_point[2 * j + 1];
            }
            printf("]\n");
            if (i == 0) {
              TDL_VisualizePolylines(point, inst_seg_meta.info[i].mask_point_size,
                                         input_image, output_image);
            } else {
              TDL_VisualizePolylines(point, inst_seg_meta.info[i].mask_point_size,
                                         input_image, output_image);
            }
        }
    }
  }

  TDL_ReleaseInstanceSegMeta(&inst_seg_meta);
  TDL_DestroyImage(image);

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
