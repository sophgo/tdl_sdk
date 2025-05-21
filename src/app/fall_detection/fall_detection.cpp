#include <algorithm>

#include <cmath>
#include "fall_detection.hpp"
#include "utils/tdl_log.hpp"

#define SCORE_THRESHOLD 0.4
#define FRAME_GAP 1.0

void print_kps(std::vector<std::pair<float, float>> &kps, int index) {
  for (uint32_t i = 0; i < kps.size(); i++) {
    printf("[%d] %d: %.2f, %.2f\n", index, i, kps[i].first, kps[i].second);
  }
}

FallDet::FallDet(uint64_t id) {
  uid = id;
  for (int i = 0; i < 4; i++) {
    valid_list.push(0);
  }

  for (int i = 0; i < 3; i++) {
    speed_caches.push(0);
  }

  for (int i = 0; i < 6; i++) {
    statuses_cache.push(0);
  }
}

int FallDet::elem_count(std::queue<int> &q) {
  int num = 0;
  int q_size = q.size();

  for (int i = 0; i < q_size; i++) {
    int val = q.front();
    q.pop();

    num += val;
    q.push(val);
  }

  return num;
}

bool FallDet::keypoints_useful(const ObjectBoxLandmarkInfo &person_meta) {
  if (history_neck.size() == FRAME_GAP + 3) {
    history_neck.erase(history_neck.begin());
    history_hip.erase(history_hip.begin());
  }

  if (person_meta.landmarks_score[5] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[6] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[11] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[12] > SCORE_THRESHOLD) {
    float neck_x =
        (person_meta.landmarks_x[5] + person_meta.landmarks_x[6]) / 2.0f;
    float neck_y =
        (person_meta.landmarks_y[5] + person_meta.landmarks_y[6]) / 2.0f;
    float hip_x =
        (person_meta.landmarks_x[11] + person_meta.landmarks_x[12]) / 2.0f;
    float hip_y =
        (person_meta.landmarks_y[11] + person_meta.landmarks_y[12]) / 2.0f;

    std::pair<float, float> neck = std::make_pair(neck_x, neck_y);
    std::pair<float, float> hip = std::make_pair(hip_x, hip_y);

    history_neck.push_back(neck);
    history_hip.push_back(hip);

    return true;

  } else {
    history_neck.push_back(std::make_pair(0, 0));
    history_hip.push_back(std::make_pair(0, 0));

    return false;
  }
}

void FallDet::get_kps(std::vector<std::pair<float, float>> &val_list, int index,
                      float *x, float *y) {
  float tmp_x = 0;
  float tmp_y = 0;

  for (int i = index; i < index + 3; i++) {
    tmp_x += val_list[i].first;
    tmp_y += val_list[i].second;
  }

  *x = tmp_x / 3.0f;
  *y = tmp_y / 3.0f;
}

float FallDet::human_orientation() {
  float neck_x, neck_y, hip_x, hip_y;

  get_kps(history_neck, FRAME_GAP, &neck_x, &neck_y);
  get_kps(history_hip, FRAME_GAP, &hip_x, &hip_y);

  float human_angle =
      atan2(hip_y - neck_y, hip_x - neck_x) * 180.0 / M_PI - 90.0;

  return human_angle;
}

float FallDet::body_box_calculation(const ObjectBoxLandmarkInfo &person_meta) {
  return (person_meta.x2 - person_meta.x1) / (person_meta.y2 - person_meta.y1);
}

void FallDet::update_queue(std::queue<int> &q, int val) {
  q.pop();
  q.push(val);
}

float FallDet::speed_detection(const ObjectBoxLandmarkInfo &person_meta,
                               float fps) {
  float neck_x_before, neck_y_before, neck_x_cur, neck_y_cur;

  get_kps(history_neck, 0, &neck_x_before, &neck_y_before);
  get_kps(history_neck, FRAME_GAP, &neck_x_cur, &neck_y_cur);

  float delta_position = sqrt(pow(neck_x_before - neck_x_cur, 2) +
                              pow(neck_y_before - neck_y_cur, 2));

  if (neck_y_cur < neck_y_before) {
    delta_position = -delta_position;
  }

  std::vector<float> delta_val;

  float box_w = person_meta.x2 - person_meta.x1;
  float box_h = person_meta.y2 - person_meta.y1;

  if (person_meta.landmarks_score[13] < SCORE_THRESHOLD &&
      person_meta.landmarks_score[14] < SCORE_THRESHOLD &&
      person_meta.landmarks_score[15] < SCORE_THRESHOLD &&
      person_meta.landmarks_score[16] < SCORE_THRESHOLD) {
    box_h *= 1.8;
  } else if (person_meta.landmarks_score[15] < SCORE_THRESHOLD &&
             person_meta.landmarks_score[16] < SCORE_THRESHOLD) {
    box_h *= 1.3;
  }

  float delta_body = sqrt(pow(box_w, 2) + pow(box_h, 2));

  delta_val.push_back(delta_body);

  if (person_meta.landmarks_score[12] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[14] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[16] > SCORE_THRESHOLD) {
    float left_leg_up =
        sqrt(pow(person_meta.landmarks_x[12] - person_meta.landmarks_x[14], 2) +
             pow(person_meta.landmarks_y[12] - person_meta.landmarks_y[14], 2));
    float left_leg_bottom =
        sqrt(pow(person_meta.landmarks_x[16] - person_meta.landmarks_x[14], 2) +
             pow(person_meta.landmarks_y[16] - person_meta.landmarks_y[14], 2));
    float left_leg = (left_leg_up + left_leg_bottom) * 2.4;

    delta_val.push_back(left_leg);
  }

  if (person_meta.landmarks_score[11] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[13] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[15] > SCORE_THRESHOLD) {
    float right_leg_up =
        sqrt(pow(person_meta.landmarks_x[11] - person_meta.landmarks_x[13], 2) +
             pow(person_meta.landmarks_y[11] - person_meta.landmarks_y[13], 2));
    float right_leg_bottom =
        sqrt(pow(person_meta.landmarks_x[13] - person_meta.landmarks_x[15], 2) +
             pow(person_meta.landmarks_y[13] - person_meta.landmarks_y[15], 2));
    float right_leg = (right_leg_up + right_leg_bottom) * 2.4;

    delta_val.push_back(right_leg);
  }

  if (person_meta.landmarks_score[6] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[8] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[10] > SCORE_THRESHOLD) {
    float left_arm_up =
        sqrt(pow(person_meta.landmarks_x[6] - person_meta.landmarks_x[8], 2) +
             pow(person_meta.landmarks_y[6] - person_meta.landmarks_y[8], 2));
    float left_arm_bottom =
        sqrt(pow(person_meta.landmarks_x[8] - person_meta.landmarks_x[10], 2) +
             pow(person_meta.landmarks_y[8] - person_meta.landmarks_y[10], 2));
    float left_arm = (left_arm_up + left_arm_bottom) * 3.4;

    delta_val.push_back(left_arm);
  }

  if (person_meta.landmarks_score[5] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[7] > SCORE_THRESHOLD &&
      person_meta.landmarks_score[9] > SCORE_THRESHOLD) {
    float right_arm_up =
        sqrt(pow(person_meta.landmarks_x[5] - person_meta.landmarks_x[7], 2) +
             pow(person_meta.landmarks_y[5] - person_meta.landmarks_y[7], 2));
    float right_arm_bottom =
        sqrt(pow(person_meta.landmarks_x[7] - person_meta.landmarks_x[9], 2) +
             pow(person_meta.landmarks_y[7] - person_meta.landmarks_y[9], 2));
    float right_arm = (right_arm_up + right_arm_bottom) * 3.4;

    delta_val.push_back(right_arm);
  }

  double delta_sum = 0.0;

  for (uint32_t i = 0; i < delta_val.size(); i++) {
    delta_sum += delta_val[i];
  }
  double delta_mean = delta_sum / (float)delta_val.size();

  float speed = 100.0 * delta_position / (delta_mean * (FRAME_GAP / fps));

  if (speed > SPEED_THRESHOLD) {
    update_queue(speed_caches, 1);
  } else {
    update_queue(speed_caches, 0);
  }

  if (elem_count(speed_caches) >= 2) {
    is_moving = true;
  } else {
    is_moving = false;
  }

  return speed;
}

int FallDet::action_analysis(float human_angle, float aspect_ratio,
                             float moving_speed) {
  /*
  state_list[0]: Stand_still
  state_list[1]: Stand_walking
  state_list[2]: Fall
  state_list[3]: Lie
  state_list[4]: Sit

  */
  float status_score[5] = {0.0};

  if (human_angle > -HUMAN_ANGLE_THRESHOLD &&
      human_angle < HUMAN_ANGLE_THRESHOLD) {
    status_score[0] += 0.8;
    status_score[1] += 0.8;
    status_score[4] += 0.8;
  } else {
    status_score[2] += 0.8;
    status_score[3] += 0.8;
  }

  if (aspect_ratio < ASPECT_RATIO_THRESHOLD) {
    status_score[0] += 0.8;
    status_score[1] += 0.8;
  } else if (aspect_ratio > 1.0f / ASPECT_RATIO_THRESHOLD) {
    status_score[3] += 0.8;
  } else {
    status_score[2] += 0.8;
    status_score[4] += 0.8;
  }

  if (moving_speed < SPEED_THRESHOLD) {
    status_score[0] += 0.8;
    status_score[1] += 0.8;
    status_score[3] += 0.8;
    status_score[4] += 0.8;
  } else {
    status_score[2] += 0.8;
  }

  if (is_moving) {
    status_score[1] += 0.8;
    status_score[2] += 0.8;
  } else {
    status_score[0] += 0.8;
    status_score[3] += 0.8;
    status_score[4] += 0.8;
  }

  int max_position =
      std::max_element(status_score, status_score + 5) - status_score;

  return max_position;
}

bool FallDet::alert_decision(int status) {
  update_queue(statuses_cache, status);

  if (elem_count(statuses_cache) >= 3) {
    return true;
  } else {
    return false;
  }
}

int FallDet::detect(const ObjectBoxLandmarkInfo &person_meta, float fps) {
  int falling = 0;
  if (keypoints_useful(person_meta)) {
    update_queue(valid_list, 1);

    if (elem_count(valid_list) == 4) {
      float human_angle = human_orientation();
      float aspect_ratio = body_box_calculation(person_meta);
      float speed = speed_detection(person_meta, fps);
      int status = action_analysis(human_angle, aspect_ratio, speed);

      int is_fall = status == 2 ? 1 : 0;

      if (alert_decision(is_fall)) {
        falling = 1;
      }
    }
  } else {
    update_queue(valid_list, 0);
  }

  return falling;
}
