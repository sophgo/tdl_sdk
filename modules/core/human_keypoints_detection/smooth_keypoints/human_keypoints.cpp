#include <algorithm>
#include "core/core/cvtdl_errno.h"
#include "core/cvi_tdl_types_mem.h"
#include "core_utils.hpp"
#include "cvi_sys.h"
#include "face_utils.hpp"

#include <cmath>
#include "cvi_tdl_log.hpp"
#include "human_keypoints.hpp"

std::pair<float, float> operator+(const std::pair<float, float>& p1,
                                  const std::pair<float, float>& p2) {
  return {p1.first + p2.first, p1.second + p2.second};
}

std::pair<float, float> operator+(const std::pair<float, float>& p, float num) {
  return {p.first + num, p.second + num};
}

std::pair<float, float> operator-(float minuend, const std::pair<float, float>& p) {
  return {minuend - p.first, minuend - p.second};
}

std::pair<float, float> operator-(const std::pair<float, float>& p1,
                                  const std::pair<float, float>& p2) {
  return {p1.first - p2.first, p1.second - p2.second};
}

std::pair<float, float> operator*(const std::pair<float, float>& p, float scalar) {
  return {p.first * scalar, p.second * scalar};
}

std::pair<float, float> operator*(const std::pair<float, float>& p1,
                                  const std::pair<float, float>& p2) {
  return {p1.first * p2.first, p1.second * p2.second};
}

std::pair<float, float> operator/(const std::pair<float, float>& p1,
                                  const std::pair<float, float>& p2) {
  if (p2.first == 0.0f || p2.second == 0.0f) {
    throw std::invalid_argument("Division by zero is not allowed.");
  }
  return {p1.first / p2.first, p1.second / p2.second};
}

std::pair<float, float> operator/(const std::pair<float, float>& p, float divisor) {
  if (divisor == 0.0f) {
    throw std::invalid_argument("Division by zero is not allowed.");
  }
  return {p.first / divisor, p.second / divisor};
}

HumanKeypoints::HumanKeypoints(int id, SmoothAlgParam smooth_param) {
  uid = id;

  image_width = smooth_param.image_width;
  image_height = smooth_param.image_height;
  fc_d = smooth_param.fc_d;
  fc_min = smooth_param.fc_min;
  beta = smooth_param.beta;
  thres_mult = smooth_param.thres_mult;
  te = smooth_param.te;
  smooth_frames = smooth_param.smooth_frames;
  smooth_type = smooth_param.smooth_type;

  if (smooth_type == 0) {
    weight_list = gen_weights(smooth_frames);

  } else if (smooth_type == 1) {
    gen_weights();
    init_alpha();
  }
}

void HumanKeypoints::gen_weights() {
  for (int i = 0; i < 17; i++) {
    if (i < 5) {
      weight_list.push_back(0.005 * thres_mult);
    } else {
      weight_list.push_back(0.01 * thres_mult);
    }
  }
}

std::vector<float> HumanKeypoints::gen_weights(int n) {
  float a1 = 1.0f / (2.0f * n);
  float d = 1.0f / (float)(n * (n - 1));

  std::vector<float> weights;
  for (int i = 0; i < n; i++) {
    weights.push_back(a1 + i * d);
    // weights.push_back(1.0f/(float)n);
  }
  return weights;
}

void HumanKeypoints::init_alpha() {
  float r = 2 * M_PI * fc_d * te;
  alpha = r / (r + 1.0f);
}

std::pair<float, float> HumanKeypoints::abs_pair(std::pair<float, float> pair) {
  float first = pair.first > 0 ? pair.first : -pair.first;
  float second = pair.second > 0 ? pair.second : -pair.second;
  return std::make_pair(first, second);
}

std::pair<float, float> HumanKeypoints::smoothing_factor(std::pair<float, float> fc_pair) {
  std::pair<float, float> r = fc_pair * 2.0f * M_PI * te;
  return r / (r + 1.0f);
}

std::pair<float, float> HumanKeypoints::exponential_smoothing(std::pair<float, float> x_cur,
                                                              std::pair<float, float> x_pre,
                                                              bool use_pair) {
  if (use_pair) {
    return x_cur * alpha_pair + x_pre * (1.0f - alpha_pair);
  } else {
    return x_cur * alpha + x_pre * (1.0f - alpha);
  }
}

std::pair<float, float> HumanKeypoints::smooth(std::pair<float, float> current_keypoint,
                                               float threshold, int index) {
  float deta_x = (current_keypoint.first - x_prev_hat[index].first) / (float)image_width;
  float deta_y = (current_keypoint.second - x_prev_hat[index].second) / (float)image_height;

  float distance = pow(deta_x * deta_x + deta_y * deta_y, 0.5);

  // printf("index:%d, distance: %f, threshold: %f, image_width:%d, image_height:%d \n", index,
  // distance, threshold, image_width,image_height );

  std::pair<float, float> result;
  if (distance < threshold) {
    result = x_prev_hat[index];
  } else {
    result = one_euro_filter(current_keypoint, x_prev_hat[index], index);
  }

  // printf("result: %f, %f\n", result.first, result.second);

  return result;
}

void HumanKeypoints::smooth_keypoints(cvtdl_pose17_meta_t* kps_meta) {
  if (smooth_type == 0) {
    weight_add(kps_meta);
    return;
  }

  if (first_time) {
    for (int i = 0; i < 17; i++) {
      x_prev_hat[i] = std::make_pair(kps_meta->x[i], kps_meta->y[i]);
    }
    first_time = false;
  } else {
    for (int i = 0; i < 17; i++) {
      std::pair<float, float> cur_keypoint = std::make_pair(kps_meta->x[i], kps_meta->y[i]);
      std::pair<float, float> smooth_keypoint = smooth(cur_keypoint, weight_list[i], i);

      kps_meta->x[i] = smooth_keypoint.first;
      kps_meta->y[i] = smooth_keypoint.second;
    }
  }
}

void HumanKeypoints::weight_add(cvtdl_pose17_meta_t* kps_meta) {
  cvtdl_pose17_meta_t kps = *kps_meta;

  history_keypoints.push_back(kps);
  size_t q_size = history_keypoints.size();

  if (q_size < 2) {
    return;
  } else if ((int)q_size > smooth_frames) {
    history_keypoints.pop_front();
  }

  q_size = history_keypoints.size();

  std::vector<float> dst_weights;

  if ((int)q_size < smooth_frames) {
    dst_weights = gen_weights(q_size);
  } else {
    dst_weights = weight_list;
  }

  for (size_t i = 0; i < q_size; ++i) {
    for (size_t j = 0; j < 17; ++j) {
      if (i == 0) {
        kps_meta->x[j] = history_keypoints[i].x[j] * dst_weights[i];
        kps_meta->y[j] = history_keypoints[i].y[j] * dst_weights[i];
      } else {
        kps_meta->x[j] += history_keypoints[i].x[j] * dst_weights[i];
        kps_meta->y[j] += history_keypoints[i].y[j] * dst_weights[i];
      }
    }
  }
}

std::pair<float, float> HumanKeypoints::one_euro_filter(std::pair<float, float> x_cur,
                                                        std::pair<float, float> x_pre, int index) {
  std::pair<float, float> dx_cur = (x_cur - x_pre) / te;
  std::pair<float, float> dx_cur_hat = exponential_smoothing(dx_cur, dx_prev_hat[index], false);

  // printf("dx_cur: %f, %f\n", dx_cur.first, dx_cur.second);
  // printf("dx_cur_hat: %f, %f\n", dx_cur_hat.first, dx_cur_hat.second);

  std::pair<float, float> fc = abs_pair(dx_cur_hat) * beta + fc_min;

  alpha_pair = smoothing_factor(fc);
  // printf("alpha_pair: %f, %f\n", alpha_pair.first, alpha_pair.second);

  std::pair<float, float> x_cur_hat = exponential_smoothing(x_cur, x_pre, true);
  // printf("x_cur_hat: %f, %f\n", x_cur_hat.first, x_cur_hat.second);

  dx_prev_hat[index] = dx_cur_hat;
  x_prev_hat[index] = x_cur_hat;
  return x_cur_hat;
}
