#pragma once

#include <deque>
#include <iostream>
#include <vector>
#include "nn/tdl_model_factory.hpp"
#include "utils/tdl_log.hpp"

typedef struct {
  uint32_t image_width = 1920;
  uint32_t image_height = 1080;
  float fc_d = 0.1;
  float fc_min = 0.1;
  float beta = 0.01;
  float thres_mult = 0.3;
  float te = 1.0f;
  ;
  int smooth_frames = 5;
  int smooth_type = 1;
} SmoothAlgParam;

class HumanKeypoints {
 public:
  HumanKeypoints(int id, SmoothAlgParam smooth_param);
  void init_alpha();
  std::pair<float, float> abs_pair(std::pair<float, float> pair);
  std::pair<float, float> smoothing_factor(std::pair<float, float> fc_pair);
  std::pair<float, float> exponential_smoothing(std::pair<float, float> x_cur,
                                                std::pair<float, float> x_pre,
                                                bool use_pair);
  std::pair<float, float> one_euro_filter(std::pair<float, float> x_cur,
                                          std::pair<float, float> x_pre,
                                          int index);
  std::pair<float, float> smooth(std::pair<float, float> current_keypoint,
                                 float threshold, int index);
  void smooth_keypoints(ObjectBoxLandmarkInfo* kps_meta);
  void weight_add(ObjectBoxLandmarkInfo* kps_meta);

  uint64_t uid;
  int MAX_UNMATCHED_TIME = 30;
  int unmatched_times = 0;

 private:
  std::vector<float> weight_list;

  // Weight add
  std::vector<float> gen_weights(int n);
  std::deque<ObjectBoxLandmarkInfo> history_keypoints;

  // OneEuro filter
  void gen_weights();
  std::vector<std::pair<float, float>> dx_prev_hat =
      std::vector<std::pair<float, float>>(17, std::make_pair(0.0f, 0.0f));
  std::vector<std::pair<float, float>> x_prev_hat =
      std::vector<std::pair<float, float>>(17, std::make_pair(0.0f, 0.0f));
  uint32_t image_width;
  uint32_t image_height;
  int smooth_frames = 5;
  int smooth_type;
  float alpha;
  float fc_d;
  float fc_min;
  float beta;
  float thres_mult;
  float te;
  std::pair<float, float> alpha_pair;
  bool first_time = true;
};