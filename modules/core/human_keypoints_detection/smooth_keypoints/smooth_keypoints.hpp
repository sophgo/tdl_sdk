#pragma once
#include "human_keypoints.hpp"

class SmoothKeypoints {
 public:
  SmoothKeypoints();
  int smooth(cvtdl_object_t* obj_meta);
  int set_smooth_frames(int num);
  SmoothAlgParam get_algparam();
  void set_algparam(SmoothAlgParam smooth_param);

 private:
  std::vector<HumanKeypoints> muti_keypoints;
  SmoothAlgParam smooth_param_;
};