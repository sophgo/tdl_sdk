#include <iomanip>
#include <iostream>
#include <vector>
#include "cv/area_detect/intrusion_detect.hpp"

// 打印点集信息的辅助函数
void printPoints(const PointsInfo& points, const std::string& title) {
  std::cout << "==== " << title << " ====" << std::endl;
  std::cout << "点数: " << points.x.size() << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  for (size_t i = 0; i < points.x.size(); i++) {
    std::cout << "点 " << i << ": (" << points.x[i] << ", " << points.y[i]
              << ")" << std::endl;
  }
  std::cout << std::endl;
}

// 注意：由于改用 vector，不再需要手动释放内存
void freeRegionInfo(std::vector<PointsInfo>& region_info) {
  region_info.clear();
}

int main() {
  std::cout << "开始测试入侵检测系统..." << std::endl;

  // 创建入侵检测对象
  IntrusionDetect detector;

  // 测试用例1: 设置凸多边形区域（矩形）
  std::cout << "测试用例1: 矩形区域" << std::endl;
  PointsInfo rectangle;
  std::vector<float> rect_x = {100.0, 300.0, 300.0, 100.0};
  std::vector<float> rect_y = {100.0, 100.0, 200.0, 200.0};
  rectangle.x = rect_x;
  rectangle.y = rect_y;

  // 打印输入区域
  printPoints(rectangle, "输入矩形区域");

  // 设置区域（添加区域名称）
  int result = detector.addRegion(rectangle, "安全区域1");
  std::cout << "addRegion 返回结果: " << result
            << (result == 0 ? " (成功)" : " (失败)") << std::endl;

  // 获取设置的区域并打印
  std::vector<PointsInfo> region_info_vector;
  detector.getRegion(region_info_vector);
  std::cout << "获取到的区域数量: " << region_info_vector.size() << std::endl;
  for (size_t i = 0; i < region_info_vector.size(); i++) {
    printPoints(region_info_vector[i], "区域 " + std::to_string(i + 1));
  }

  // 测试入侵检测 - 在区域内部的矩形
  ObjectBoxInfo bbox_inside(0, 0.9, 150.0, 120.0, 200.0, 180.0);

  bool is_intruding = detector.isIntrusion(bbox_inside);
  std::cout << "内部边界框入侵检测结果: " << (is_intruding ? "入侵" : "未入侵")
            << std::endl;

  // 测试入侵检测 - 在区域外部的矩形
  ObjectBoxInfo bbox_outside(0, 0.9, 50.0, 50.0, 80.0, 80.0);

  is_intruding = detector.isIntrusion(bbox_outside);
  std::cout << "外部边界框入侵检测结果: " << (is_intruding ? "入侵" : "未入侵")
            << std::endl;

  // 测试用例2: 设置非凸多边形区域（凹多边形）
  std::cout << "\n测试用例2: 凹多边形区域" << std::endl;
  PointsInfo concave;
  std::vector<float> concave_x = {100.0, 200.0, 300.0, 250.0, 200.0, 150.0};
  std::vector<float> concave_y = {100.0, 50.0, 100.0, 150.0, 120.0, 150.0};
  concave.x = concave_x;
  concave.y = concave_y;

  // 打印输入区域
  printPoints(concave, "输入凹多边形区域");

  // 设置区域（添加区域名称）
  result = detector.addRegion(concave, "危险区域A");
  region_info_vector.clear();
  detector.getRegion(region_info_vector);
  std::cout << "addRegion 返回结果: " << result
            << (result == 0 ? " (成功)" : " (失败)") << std::endl;

  // 获取设置的区域并打印
  std::cout << "获取到的区域数量（应该大于1，因为是凹多边形）: "
            << region_info_vector.size() << std::endl;
  for (size_t i = 0; i < region_info_vector.size(); i++) {
    printPoints(region_info_vector[i],
                "分解后的凸多边形 " + std::to_string(i + 1));
  }

  // 测试入侵检测 - 在凹区域内部的矩形
  ObjectBoxInfo bbox_inside_concave(0, 0.9, 200.0, 100.0, 220.0, 120.0);

  is_intruding = detector.isIntrusion(bbox_inside_concave);
  std::cout << "凹多边形区域内部边界框入侵检测结果: "
            << (is_intruding ? "入侵" : "未入侵") << std::endl;

  // 测试入侵检测 - 在凹区域外部的矩形
  ObjectBoxInfo bbox_outside_concave(0, 0.9, 300.0, 300.0, 420.0, 420.0);

  is_intruding = detector.isIntrusion(bbox_outside_concave);
  std::cout << "凹多边形区域外部边界框入侵检测结果: "
            << (is_intruding ? "入侵" : "未入侵") << std::endl;

  // 测试默认区域名称
  PointsInfo default_region;
  default_region.x = {0.0, 100.0, 100.0, 0.0};
  default_region.y = {0.0, 0.0, 100.0, 100.0};
  result = detector.addRegion(default_region);  // 使用默认名称
  std::cout << "添加默认名称区域结果: " << result
            << (result == 0 ? " (成功)" : " (失败)") << std::endl;

  // 清理detector
  detector.clean();

  std::cout << "入侵检测系统测试完成！" << std::endl;
  return 0;
}