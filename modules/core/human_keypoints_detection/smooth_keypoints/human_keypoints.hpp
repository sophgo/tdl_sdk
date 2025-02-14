#pragma once

#include <deque>
#include <iostream>
#include <vector>
#include "core/object/cvtdl_object_types.h"

class HumanKeypoints {
 public:
  HumanKeypoints(int id, SmoothAlgParam smooth_param);
  void init_alpha();
  std::pair<float, float> abs_pair(std::pair<float, float> pair);
  std::pair<float, float> smoothing_factor(std::pair<float, float> fc_pair);
  std::pair<float, float> exponential_smoothing(std::pair<float, float> x_cur,
                                                std::pair<float, float> x_pre, bool use_pair);
  std::pair<float, float> one_euro_filter(std::pair<float, float> x_cur,
                                          std::pair<float, float> x_pre, int index);
  std::pair<float, float> smooth(std::pair<float, float> current_keypoint, float threshold,
                                 int index);
  void smooth_keypoints(cvtdl_pose17_meta_t* kps_meta);
  void weight_add(cvtdl_pose17_meta_t* kps_meta);

  int uid;
  int MAX_UNMATCHED_TIME = 30;
  int unmatched_times = 0;

 private:
  int smooth_type;
  std::vector<float> weight_list;

  // Weight add
  std::vector<float> gen_weights(int n);
  std::deque<cvtdl_pose17_meta_t> history_keypoints;
  int smooth_frames = 5;

  // OneEuro filter
  void gen_weights();
  std::vector<std::pair<float, float>> dx_prev_hat =
      std::vector<std::pair<float, float>>(17, std::make_pair(0.0f, 0.0f));
  std::vector<std::pair<float, float>> x_prev_hat =
      std::vector<std::pair<float, float>>(17, std::make_pair(0.0f, 0.0f));
  uint32_t image_width;
  uint32_t image_height;
  float alpha;
  float fc_d;
  float fc_min;
  float beta;
  float thres_mult;
  float te;
  std::pair<float, float> alpha_pair;
  bool first_time = true;
};