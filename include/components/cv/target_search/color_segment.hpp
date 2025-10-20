#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "image/base_image.hpp"
#include "preprocess/base_preprocessor.hpp"
struct cvtdl_color_params_t {
  float expand_h_ratio = 0.15;  // 种子点扩展比例（控制裁剪区域大小）
  int patch_radius = 2;         // 计算颜色均值的ROI半径
  int min_area = 25;  // 最小目标面积（过滤小连通域噪声）
  std::vector<int> diff_list = {10, 15, 20};  // 颜色差异阈值列表（优先小阈值）
};

struct cvtdl_color_result_t {
  cv::Rect bbox;    // 目标边界框（原图坐标系）
  cv::Mat fg_mask;  // 前景掩码（原图尺寸，0=背景，255=目标）
  cv::Mat result_mask;   // 结果掩码（与fg_mask一致，兼容接口）
  bool success = false;  // 处理成功标志（true=找到有效目标）
};

class ColorSegmentor {
 public:
  ColorSegmentor();
  explicit ColorSegmentor(const cvtdl_color_params_t& params);

  int segment(std::shared_ptr<BaseImage> image, cv::Point seed_point,
              cvtdl_color_result_t* result);

  void setParams(const cvtdl_color_params_t& params);

 private:
  cvtdl_color_params_t params_;
  bool in_center(const cv::Rect& bbox, const cv::Point& seed);
  std::shared_ptr<BasePreprocessor> preprocessor;
};