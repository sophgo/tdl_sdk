#pragma once
#include "core/object/cvai_object_types.h"

// #include <deque>
#include <iostream>
#include <queue>

class FallDet {
 public:
  FallDet(int id);
  void detect(cvai_object_info_t* meta, float fps);
  void update_queue(std::queue<int>& q, int val);
  std::queue<int> valid_list;

  //   void set_speed(int speed);

  int uid;
  int unmatched_times = 0;
  int MAX_UNMATCHED_TIME = 30;

 private:
  void get_kps(std::vector<std::pair<float, float>>& val_list, int index, float* x, float* y);
  float human_orientation();
  bool keypoints_useful(cvai_pose17_meta_t* kps_meta);
  float body_box_calculation(cvai_bbox_t* bbox);
  float speed_detection(cvai_bbox_t* bbox, cvai_pose17_meta_t* kps_meta, float fps);
  int elem_count(std::queue<int>& q);
  //   void update_queue(std::queue<int>& q, int val);
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