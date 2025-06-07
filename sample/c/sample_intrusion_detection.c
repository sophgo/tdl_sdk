#include <stdio.h>
#include <stdlib.h>
#include "tdl_sdk.h"

// 打印点集信息的辅助函数
void print_points(const TDLPoints* points, const char* title) {
  printf("==== %s ====\n", title);
  printf("点数: %d\n", points->size);
  for (uint32_t i = 0; i < points->size; i++) {
    printf("点 %d: (%.2f, %.2f)\n", i, points->x[i], points->y[i]);
  }
  printf("\n");
}

int main() {
  TDLHandle handle = TDL_CreateHandle(0);
  bool is_intrusion = false;

  printf("=============矩形区域测试=============\n");

  // 创建矩形区域点集
  TDLPoints rectangle;
  rectangle.size = 4;
  rectangle.x = (float*)malloc(sizeof(float) * rectangle.size);
  rectangle.y = (float*)malloc(sizeof(float) * rectangle.size);

  // 设置矩形顶点坐标
  rectangle.x[0] = 100.0f;
  rectangle.y[0] = 100.0f;
  rectangle.x[1] = 300.0f;
  rectangle.y[1] = 100.0f;
  rectangle.x[2] = 300.0f;
  rectangle.y[2] = 200.0f;
  rectangle.x[3] = 100.0f;
  rectangle.y[3] = 200.0f;

  print_points(&rectangle, "矩形区域");

  // 测试入侵检测 - 在区域内部的矩形
  TDLBox bbox_inside = {.x1 = 150.0f, .y1 = 120.0f, .x2 = 200.0f, .y2 = 180.0f};

  TDL_IntrusionDetection(handle, &rectangle, &bbox_inside, &is_intrusion);
  printf("内部边界框检测: %s\n", is_intrusion ? "入侵" : "未入侵");

  // 测试入侵检测 - 在区域外部的矩形
  TDLBox bbox_outside = {.x1 = 50.0f, .y1 = 50.0f, .x2 = 80.0f, .y2 = 80.0f};

  TDL_IntrusionDetection(handle, &rectangle, &bbox_outside, &is_intrusion);
  printf("外部边界框检测: %s\n\n", is_intrusion ? "入侵" : "未入侵");

  printf("=============凹多边形区域测试=============\n");

  TDLPoints concave;
  concave.size = 6;
  concave.x = (float*)malloc(sizeof(float) * concave.size);
  concave.y = (float*)malloc(sizeof(float) * concave.size);

  concave.x[0] = 100.0f;
  concave.y[0] = 100.0f;
  concave.x[1] = 200.0f;
  concave.y[1] = 50.0f;
  concave.x[2] = 300.0f;
  concave.y[2] = 100.0f;
  concave.x[3] = 250.0f;
  concave.y[3] = 150.0f;
  concave.x[4] = 200.0f;
  concave.y[4] = 120.0f;
  concave.x[5] = 150.0f;
  concave.y[5] = 150.0f;

  print_points(&concave, "凹多边形区域");

  // 测试入侵检测 - 在凹区域内部的矩形
  TDLBox bbox_inside_concave = {
      .x1 = 200.0f, .y1 = 100.0f, .x2 = 220.0f, .y2 = 120.0f};

  TDL_IntrusionDetection(handle, &concave, &bbox_inside_concave, &is_intrusion);
  printf("内部边界框检测: %s\n", is_intrusion ? "入侵" : "未入侵");

  // 测试入侵检测 - 在凹区域外部的矩形
  TDLBox bbox_outside_concave = {
      .x1 = 300.0f, .y1 = 300.0f, .x2 = 420.0f, .y2 = 420.0f};

  TDL_IntrusionDetection(handle, &concave, &bbox_outside_concave,
                         &is_intrusion);
  printf("外部边界框检测: %s\n\n", is_intrusion ? "入侵" : "未入侵");

  // 释放资源
  free(rectangle.x);
  free(rectangle.y);
  free(concave.x);
  free(concave.y);
  TDL_DestroyHandle(handle);
  printf("测试完成！\n");
  return 0;
}