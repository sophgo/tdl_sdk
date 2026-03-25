#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "meta_visualize.h"
#include "tdl_sdk.h"

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf(
      "  %s -m <model_path> -i <input1> -j <input2> [-k <input3>] [-o "
      "<output_image>]\n",
      prog_name);
  printf(
      "  %s --model_path <path> --input1 <image> --input2 <image> [--input3 "
      "<image>] [--output <image>]\n\n",
      prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Path to cvimodel eg. <topformer_seg_motion>\n");
  printf("  -i, --input1      Path to input image 1\n");
  printf("  -j, --input2      Path to input image 2\n");
  printf("  -k, --input3      Path to input image 3 (optional)\n");
  printf("  -o, --output      Path to output image\n");
  printf(
      "  -a, --min_area    Minimum connected-component area (default: 256)\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *input_images[3] = {NULL, NULL, NULL};
  char *output_image = NULL;
  uint32_t min_area = 256;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"input1", required_argument, 0, 'i'},
                                  {"input2", required_argument, 0, 'j'},
                                  {"input3", required_argument, 0, 'k'},
                                  {"output", required_argument, 0, 'o'},
                                  {"min_area", required_argument, 0, 'a'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:i:j:k:o:a:h", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'i':
        input_images[0] = optarg;
        break;
      case 'j':
        input_images[1] = optarg;
        break;
      case 'k':
        input_images[2] = optarg;
        break;
      case 'o':
        output_image = optarg;
        break;
      case 'a':
        min_area = (uint32_t)strtoul(optarg, NULL, 10);
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

  if (!model_path || !input_images[0] || !input_images[1]) {
    fprintf(stderr,
            "Error: model_path and at least two input images are required\n");
    print_usage(argv[0]);
    return -1;
  }

  int num_images = (input_images[2] != NULL) ? 3 : 2;

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Input image1:  %s\n", input_images[0]);
  printf("  Input image2:  %s\n", input_images[1]);
  if (num_images == 3) {
    printf("  Input image3:  %s\n", input_images[2]);
  }
  printf("  Output image:  %s\n", output_image);
  printf("  min_area:      %u\n", min_area);

  int ret = 0;
  TDLModel model_id = TDL_MODEL_TOPFORMER_SEG_MOTION;
  TDLHandle tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    printf("TDL_CreateHandle failed\n");
    return -1;
  }

  ret = TDL_OpenModel(tdl_handle, model_id, model_path, NULL, 0);
  if (ret != 0) {
    printf("open model failed with %#x!\n", ret);
    goto exit0;
  }

  for (int idx = 0; idx < num_images; idx++) {
    TDLImage image = TDL_ReadImageGray(input_images[idx]);
    if (image == NULL) {
      printf("read image%d failed with %#x!\n", idx + 1, ret);
      ret = -1;
      goto exit1;
    }

    TDLObject obj_meta = {0};
    ret = TDL_SegMotionDetection(tdl_handle, model_id, image, min_area,
                                 &obj_meta);
    if (ret != 0) {
      printf("TDL_SegMotionDetection failed with %#x!\n", ret);
    } else {
      if (obj_meta.size <= 0) {
        printf("image%d: No motion detected\n", idx + 1);
      } else {
        box_t boxes[obj_meta.size];
        for (int i = 0; i < obj_meta.size; i++) {
          printf("image%d obj_meta_index : %d, bbox : [%f %f %f %f]\n", idx + 1,
                 i, obj_meta.info[i].box.x1, obj_meta.info[i].box.y1,
                 obj_meta.info[i].box.x2, obj_meta.info[i].box.y2);
          boxes[i].x1 = obj_meta.info[i].box.x1;
          boxes[i].y1 = obj_meta.info[i].box.y1;
          boxes[i].x2 = obj_meta.info[i].box.x2;
          boxes[i].y2 = obj_meta.info[i].box.y2;
        }
        if (output_image != NULL) {
          char output_path[512];
          const char *dot = strrchr(output_image, '.');
          const char *slash = strrchr(output_image, '/');
          int has_ext = dot != NULL && (slash == NULL || dot > slash);
          if (has_ext) {
            int prefix_len = (int)(dot - output_image);
            snprintf(output_path, sizeof(output_path), "%.*s_%d%s", prefix_len,
                     output_image, idx + 1, dot);
          } else {
            snprintf(output_path, sizeof(output_path), "%s_%d", output_image,
                     idx + 1);
          }
          VisualizeRectangleFromFile(boxes, obj_meta.size, input_images[idx],
                                     output_path);
        }
      }
    }

    TDL_ReleaseObjectMeta(&obj_meta);
    TDL_DestroyImage(image);
  }

exit1:
  TDL_CloseModel(tdl_handle, model_id);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
