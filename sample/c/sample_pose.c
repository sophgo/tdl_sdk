#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "meta_visualize.h"

static int skeleton[19][2] = {{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                       {5, 11},  {6, 12},  {5, 6},   {5, 7},   {6, 8},
                       {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},
                       {1, 3},   {2, 4},   {3, 5},   {4, 6}};

void print_usage(const char *prog_name) {
    printf("Usage:\n");
    printf("  %s -m <model_path> -i <input_image> -o <output_image>\n", prog_name);
    printf("  %s --model_path <path> --input <image> --output <image>\n\n", prog_name);
    printf("Options:\n");
    printf("  -m, --model_path     Path to model"
           "<keypoint_yolov8pose_person17_xxx>\n");
    printf("  -i, --input          Path to input image\n");
    printf("  -o, --output         Path to output image\n");
    printf("  -h, --help           Show this help message\n");
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

  if (!model_path || !input_image) {
    fprintf(stderr, "Error: model_path and input_image are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image:   %s\n", input_image);
  printf("  Output image:  %s\n", output_image);

  int ret = 0;

  TDLModel model_id = TDL_MODEL_KEYPOINT_YOLOV8POSE_PERSON17;
  TDLHandle tdl_handle = TDL_CreateHandle(0);

  ret = TDL_OpenModel(tdl_handle, model_id, model_path);
  if (ret != 0) {
    printf("open pose model failed with %#x!\n", ret);
    goto exit0;
  }

  //The default threshold is 0.5
  ret = TDL_SetModelThreshold(tdl_handle, model_id, 0.5);
  if (ret != 0) {
    printf("TDL_SetModelThreshold failed with %#x!\n", ret);
    goto exit1;
  }

  TDLImage image = TDL_ReadImage(input_image);
  if (image == NULL) {
    printf("read image failed with %#x!\n", ret);
    goto exit1;
  }

  TDLObject obj_meta = {0};
  ret = TDL_Detection(tdl_handle, model_id, image, &obj_meta);
  if (ret != 0) {
    printf("TDL_Pose failed with %#x!\n", ret);
  } else {
    if (obj_meta.size <= 0) {
        printf("None to detection\n");
    } else {
      point_t point[obj_meta.size * 17];
      box_t line[19];
      for (int i = 0; i < obj_meta.size; i++) {
          printf("obj_meta_index : %d, ", i);
          printf("class_id : %d, ", obj_meta.info[i].class_id);
          printf("score : %f, ", obj_meta.info[i].score);
          printf("bbox : [%f %f %f %f]\n", obj_meta.info[i].box.x1,
                                            obj_meta.info[i].box.x2,
                                            obj_meta.info[i].box.y1,
                                            obj_meta.info[i].box.y2);
          for (int j = 0; j < 17; j ++) {
            printf("pose : %d: %f %f %f\n", j, obj_meta.info[i].landmark_properity[j].x,
                obj_meta.info[i].landmark_properity[j].y,
                obj_meta.info[i].landmark_properity[j].score);

            if (obj_meta.info[i].landmark_properity[j].score < 0.5) continue;
            point[i * 17 + j].x = obj_meta.info[i].landmark_properity[j].x;
            point[i * 17 + j].y = obj_meta.info[i].landmark_properity[j].y;
          }

          for (int k = 0; k < 19; k ++) {
            int kps1 = skeleton[k][0];
            int kps2 = skeleton[k][1];

            if (obj_meta.info[i].landmark_properity[kps1].score < 0.5 ||
                obj_meta.info[i].landmark_properity[kps2].score < 0.5) {
              continue;
            }

            line[k].x1 = obj_meta.info[i].landmark_properity[kps1].x;
            line[k].y1 = obj_meta.info[i].landmark_properity[kps1].y;
            line[k].x2 = obj_meta.info[i].landmark_properity[kps2].x;
            line[k].y2 = obj_meta.info[i].landmark_properity[kps2].y;
          }
      }
      if (output_image != NULL) {
        TDL_VisualizePoint(point, obj_meta.size * 17, input_image, output_image);
        TDL_VisualizeLine(line, 19, input_image, output_image);
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
