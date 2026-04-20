#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "image/base_image.hpp"
#include "preprocess/base_preprocessor.hpp"

struct cvtdl_grabcut_params_t {
  int iter_count = 8;
  float expand_h_ratio = 0.15;
};

struct cvtdl_grabcut_result_t {
  cv::Rect bbox;
  cv::Mat fg_mask;
  cv::Mat result_mask;
  bool success = false;
};

class GrabCutSegmentor {
 public:
  GrabCutSegmentor();
  explicit GrabCutSegmentor(const cvtdl_grabcut_params_t& params);

  int segment(std::shared_ptr<BaseImage> image, cv::Point seed_point,
              cvtdl_grabcut_result_t* result);

  void setParams(const cvtdl_grabcut_params_t& params);

 private:
  cvtdl_grabcut_params_t params_;
  std::shared_ptr<BasePreprocessor> preprocessor;
};