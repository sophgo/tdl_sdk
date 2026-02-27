#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "image/base_image.hpp"
#include "tdl_model_factory.hpp"

struct cvtdl_fastsam_result_t {
  cv::Rect bbox;
  bool success = false;
};

class FastSAMSegmentor {
 public:
  FastSAMSegmentor() = default;
  explicit FastSAMSegmentor(const std::string& model_path);

  int segment(std::shared_ptr<BaseImage> image, cv::Point seed_point,
              cvtdl_fastsam_result_t* result);

 private:
  std::string model_path_;
  std::shared_ptr<BaseModel> model_od_;
};
