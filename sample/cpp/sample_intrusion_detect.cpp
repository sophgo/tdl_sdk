#include <iomanip>
#include <iostream>
#include <vector>
#include "cv/intrusion_detect/intrusion_detect.hpp"

// 打印点集信息的辅助函数
void printPoints(const PointsInfo& points, const std::string& title) {
  printf("==== %s ====\n", title.c_str());
  printf("点数: %zu\n", points.x.size());
  printf("%.2f\n", points.x[0]);
  for (size_t i = 0; i < points.x.size(); i++) {
    printf("点 %zu: (%.2f, %.2f)\n", i, points.x[i], points.y[i]);
  }
  printf("\n");
}

int main() {
  IntrusionDetection detector;

  printf("=============矩形区域测试=============\n");
  PointsInfo rectangle;
  std::vector<float> rect_x = {100.0, 300.0, 300.0, 100.0};
  std::vector<float> rect_y = {100.0, 100.0, 200.0, 200.0};
  rectangle.x = rect_x;
  rectangle.y = rect_y;

  // 打印输入区域
  printPoints(rectangle, "矩形区域");

  // 设置区域（添加区域名称）
  int result = detector.addRegion(rectangle, "安全区域1");

  std::vector<PointsInfo> region_info_vector;
  detector.getRegion(region_info_vector);
  printf("获取到的区域数量: %zu\n", region_info_vector.size());
  for (size_t i = 0; i < region_info_vector.size(); i++) {
    printPoints(region_info_vector[i], "区域 " + std::to_string(i + 1));
  }

  // 测试入侵检测 - 在区域内部的矩形
  ObjectBoxInfo bbox_inside(0, 0.9, 150.0, 120.0, 200.0, 180.0);

  result = detector.isIntrusion(bbox_inside);
  printf("内部边界框检测: %s\n", result == 1 ? "入侵" : "未入侵");

  // 测试入侵检测 - 在区域外部的矩形
  ObjectBoxInfo bbox_outside(0, 0.9, 50.0, 50.0, 80.0, 80.0);

  result = detector.isIntrusion(bbox_outside);
  printf("外部边界框检测: %s\n\n", result == 1 ? "入侵" : "未入侵");

  // 测试用例2: 设置非凸多边形区域（凹多边形）
  printf("=============凹多边形区域测试=============\n");
  PointsInfo concave;
  std::vector<float> concave_x = {100.0, 200.0, 300.0, 250.0, 200.0, 150.0};
  std::vector<float> concave_y = {100.0, 50.0, 100.0, 150.0, 120.0, 150.0};
  concave.x = concave_x;
  concave.y = concave_y;

  // 打印输入区域
  printPoints(concave, "凹多边形区域");

  // 设置区域（添加区域名称）
  result = detector.addRegion(concave, "危险区域A");
  region_info_vector.clear();
  detector.getRegion(region_info_vector);

  for (size_t i = 0; i < region_info_vector.size(); i++) {
    printPoints(region_info_vector[i],
                "分解后的凸多边形 " + std::to_string(i + 1));
  }

  // 测试入侵检测 - 在凹区域内部的矩形
  ObjectBoxInfo bbox_inside_concave(0, 0.9, 200.0, 100.0, 220.0, 120.0);

  result = detector.isIntrusion(bbox_inside_concave);
  printf("内部边界框检测: %s\n", result == 1 ? "入侵" : "未入侵");

  // 测试入侵检测 - 在凹区域外部的矩形
  ObjectBoxInfo bbox_outside_concave(0, 0.9, 300.0, 300.0, 420.0, 420.0);

  result = detector.isIntrusion(bbox_outside_concave);
  printf("外部边界框检测: %s\n\n", result == 1 ? "入侵" : "未入侵");

  // 测试默认区域名称
  PointsInfo default_region;
  default_region.x = {0.0, 100.0, 100.0, 0.0};
  default_region.y = {0.0, 0.0, 100.0, 100.0};
  result = detector.addRegion(default_region);  // 使用默认名称
  printf("=============默认区域名称测试=============\n");
  printf("添加默认名称区域结果: %s\n\n", result == 0 ? "成功" : "失败");

  // 清理detector
  detector.clean();

  return 0;
}