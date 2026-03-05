#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "tdl_sdk.h"
#include "tdl_utils.h"
#include "utils/meta_visualize.h"

// 保存原始深度数据到文本文件
void save_depth_to_txt(TDLDepthLogits *depth_logits, const char *filename) {
  int width = depth_logits->w;
  int height = depth_logits->h;
  float *logits = depth_logits->logits;

  FILE *fp = fopen(filename, "w");
  if (!fp) {
    printf("Failed to create depth data file: %s\n", filename);
    return;
  }

  // 写入深度图尺寸
  fprintf(fp, "%d %d\n", width, height);

  // 写入深度数据
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int index = y * width + x;
      fprintf(fp, "%f ", logits[index]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  printf("Depth data saved to: %s\n", filename);
}

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s -m <model_path> -l <left_image> -r <right_image>\n", prog_name);
  printf("  %s --model_path <path> --left <image> --right <image>\n\n",
         prog_name);
  printf("Options:\n");
  printf("  -m, --model_path  Path to stereo depth model\n");
  printf("  -l, --left        Path to left input image\n");
  printf("  -r, --right       Path to right input image\n");
  printf("  -h, --help        Show this help message\n");
}

int main(int argc, char *argv[]) {
  char *model_path = NULL;
  char *left_image = NULL;
  char *right_image = NULL;

  struct option long_options[] = {{"model_path", required_argument, 0, 'm'},
                                  {"left", required_argument, 0, 'l'},
                                  {"right", required_argument, 0, 'r'},
                                  {"help", no_argument, 0, 'h'},
                                  {NULL, 0, NULL, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "m:l:r:h", long_options, NULL)) != -1) {
    switch (opt) {
      case 'm':
        model_path = optarg;
        break;
      case 'l':
        left_image = optarg;
        break;
      case 'r':
        right_image = optarg;
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
  if (!model_path || !left_image || !right_image) {
    fprintf(stderr,
            "Error: model_path, left_image and right_image are required\n");
    print_usage(argv[0]);
    return -1;
  }

  printf("Running with:\n");
  printf("  Model path:    %s\n", model_path);
  printf("  Left image:    %s\n", left_image);
  printf("  Right image:   %s\n", right_image);

  int ret = 0;

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  if (!tdl_handle) {
    printf("Create handle failed!\n");
    return -1;
  }

  ret = TDL_OpenModel(tdl_handle, TDL_MODEL_DEPTH_ESTIMATION_STEREO, model_path,
                      NULL, 0);
  if (ret != 0) {
    printf("Open model failed with %#x!\n", ret);
    goto exit0;
  }

  // 读取左右图像
  TDLImage left_img = TDL_ReadImage(left_image);
  if (left_img == NULL) {
    printf("Read left image failed!\n");
    goto exit1;
  }

  TDLImage right_img = TDL_ReadImage(right_image);
  if (right_img == NULL) {
    printf("Read right image failed!\n");
    TDL_DestroyImage(left_img);
    goto exit1;
  }

  // 准备深度估计结果结构
  TDLDepthLogits depth_logits = {0};

  // 执行深度估计
  ret = TDL_DepthStereo(tdl_handle, TDL_MODEL_DEPTH_ESTIMATION_STEREO, left_img,
                        right_img, &depth_logits);
  if (ret != 0) {
    printf("TDL_DepthStereo failed with %#x!\n", ret);
  } else {
    printf("Depth estimation successful!\n");
    printf("Depth map size: %d x %d\n", depth_logits.w, depth_logits.h);
    if (depth_logits.logits) {
      printf("Depth logits data available\n");

      save_depth_to_txt(&depth_logits, "stereo_depth_data.txt");

      // 可视化深度图
      VisualizeDepthMap(&depth_logits, "stereo_depth.png");

      // 释放深度估计结果
      free(depth_logits.logits);
    }
  }

  // 释放资源
  TDL_DestroyImage(left_img);
  TDL_DestroyImage(right_img);

exit1:
  TDL_CloseModel(tdl_handle, TDL_MODEL_DEPTH_ESTIMATION_STEREO);

exit0:
  TDL_DestroyHandle(tdl_handle);
  return ret;
}
