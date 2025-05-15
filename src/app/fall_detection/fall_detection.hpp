#ifndef FALL_DETECTION_HPP
#define FALL_DETECTION_HPP

#include "common/model_output_types.hpp"

#include <iostream>
#include <queue>

class FallDet {
 public:
  FallDet(uint64_t id);
  int detect(const ObjectBoxLandmarkInfo &person_meta, float fps);
  void update_queue(std::queue<int>& q, int val);
  std::queue<int> valid_list;

  uint64_t uid;
  int unmatched_times = 0;
  int MAX_UNMATCHED_TIME = 30;

 private:
  void get_kps(std::vector<std::pair<float, float>>& val_list, int index, float* x, float* y);
  float human_orientation();
  bool keypoints_useful(const ObjectBoxLandmarkInfo &person_meta);
  float body_box_calculation(const ObjectBoxLandmarkInfo &person_meta);
  float speed_detection(const ObjectBoxLandmarkInfo &person_meta, float fps);
  int elem_count(std::queue<int>& q);

  int action_analysis(float human_angle, float aspect_ratio, float moving_speed);
  bool alert_decision(int status);

  std::vector<std::pair<float, float>> history_neck;
  std::vector<std::pair<float, float>> history_hip;
  std::queue<int> speed_caches;
  std::queue<int> statuses_cache;

  bool is_moving;
  float SPEED_THRESHOLD = 95.0;

  float HUMAN_ANGLE_THRESHOLD = 25.0;
  float ASPECT_RATIO_THRESHOLD = 0.6;
};

#endif
