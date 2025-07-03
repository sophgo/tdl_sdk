#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tdl_sdk.h"

void print_usage(const char *prog_name) {
  printf("Usage:\n");
  printf("  %s <background_image> <detect_image>\n", prog_name);
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    print_usage(argv[0]);
    return -1;
  }

  const char *background_image_path = argv[1];
  const char *detect_image_path = argv[2];

  printf("Running motion detection with images:\n");
  printf("Background: %s\n", background_image_path);
  printf("Detect: %s\n", detect_image_path);

  TDLHandle tdl_handle = TDL_CreateHandle(0);
  if (tdl_handle == NULL) {
    printf("TDL_CreateHandle failed\n");
    return -1;
  }

  TDLImage background_image = TDL_ReadImage(background_image_path);
  TDLImage detect_image = TDL_ReadImage(detect_image_path);
  // 设置ROI区域
  printf("Setting ROI regions...\n");
  TDLObject roi = {0};
  roi.size = 2;
  roi.info = (TDLObjectInfo *)malloc(roi.size * sizeof(TDLObjectInfo));

  // 设置第一个ROI区域
  roi.info[0].box.x1 = 0;
  roi.info[0].box.y1 = 0;
  roi.info[0].box.x2 = 512;
  roi.info[0].box.y2 = 512;

  // 设置第二个ROI区域
  roi.info[1].box.x1 = 1000;
  roi.info[1].box.y1 = 150;
  roi.info[1].box.x2 = 1150;
  roi.info[1].box.y2 = 250;
  TDLObject obj_meta = {0};
  printf("Begin motion detection...\n");
  TDL_MotionDetection(tdl_handle, background_image, detect_image, &roi, 20, 50,
                      &obj_meta, 0);
  free(roi.info);
  TDL_DestroyImage(background_image);
  TDL_DestroyImage(detect_image);
  for (size_t i = 0; i < obj_meta.size; i++) {
    printf("[%d,%d,%d,%d]\n", (int)obj_meta.info[i].box.x1,
           (int)obj_meta.info[i].box.y1, (int)obj_meta.info[i].box.x2,
           (int)obj_meta.info[i].box.y2);
  }

  // 清理资源
  TDL_DestroyHandle(tdl_handle);

  printf("Detection completed\n");
  return 0;
}