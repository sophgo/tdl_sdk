#include "kalman_box_tracker.hpp"
#include <algorithm>
#include <cmath>

KalmanBoxTracker::KalmanBoxTracker(const std::vector<float>& bbox) {
  // 初始化状态向量和相关矩阵
  // 状态向量：[x, y, w, h, vx, vy, vw, vh]

  // 初始化测量矩阵 H: 从状态空间到测量空间 [x, y, w, h]
  measurement_matrix_ = Eigen::MatrixXf::Zero(4, 8);
  measurement_matrix_(0, 0) = 1.0f;  // x
  measurement_matrix_(1, 1) = 1.0f;  // y
  measurement_matrix_(2, 2) = 1.0f;  // w
  measurement_matrix_(3, 3) = 1.0f;  // h

  // 初始化状态转移矩阵 F: [x, y, w, h, vx, vy, vw, vh] -> [x+vx, y+vy, w+vw,
  // h+vh, vx, vy, vw, vh]
  transition_matrix_ = Eigen::MatrixXf::Identity(8, 8);
  transition_matrix_(0, 4) = 1.0f;  // x += vx
  transition_matrix_(1, 5) = 1.0f;  // y += vy
  transition_matrix_(2, 6) = 1.0f;  // w += vw
  transition_matrix_(3, 7) = 1.0f;  // h += vh

  // 初始化过程噪声协方差
  process_noise_cov_ = Eigen::MatrixXf::Identity(8, 8) * 0.001f;

  // 初始化测量噪声协方差
  measurement_noise_cov_ = Eigen::MatrixXf::Identity(4, 4) * 0.2f;
  base_measurement_noise_cov_ = measurement_noise_cov_;

  // 初始化后验误差协方差
  error_cov_post_ = Eigen::MatrixXf::Identity(8, 8);

  // 初始化状态向量 [x, y, w, h, vx, vy, vw, vh]
  float x = bbox[0];
  float y = bbox[1];
  float w = bbox[2];
  float h = bbox[3];
  float cx = x + w / 2.0f;
  float cy = y + h / 2.0f;

  state_post_ = Eigen::MatrixXf::Zero(8, 1);
  state_post_(0, 0) = cx;
  state_post_(1, 0) = cy;
  state_post_(2, 0) = w;
  state_post_(3, 0) = h;
  // 速度初始化为0
  state_post_(4, 0) = 0.0f;
  state_post_(5, 0) = 0.0f;
  state_post_(6, 0) = 0.0f;
  state_post_(7, 0) = 0.0f;

  // 记录最后一次可信的尺度
  last_reliable_width_ = w;
  last_reliable_height_ = h;

  // 初始化计数器
  update_count_ = 1;
}

void KalmanBoxTracker::update(const std::vector<float>& bbox,
                              bool update_scale) {
  float x = bbox[0];
  float y = bbox[1];
  float w = bbox[2];
  float h = bbox[3];
  float cx = x + w / 2.0f;
  float cy = y + h / 2.0f;

  if (update_scale) {
    // 使用检测框更新时，更新位置和尺度
    Eigen::MatrixXf measurement(4, 1);
    measurement(0, 0) = cx;
    measurement(1, 0) = cy;
    measurement(2, 0) = w;
    measurement(3, 0) = h;

    // 卡尔曼滤波更新步骤
    Eigen::MatrixXf innovation = measurement - measurement_matrix_ * state_pre_;
    Eigen::MatrixXf innovation_covariance =
        measurement_matrix_ * error_cov_pre_ * measurement_matrix_.transpose() +
        measurement_noise_cov_;
    Eigen::MatrixXf kalman_gain = error_cov_pre_ *
                                  measurement_matrix_.transpose() *
                                  innovation_covariance.inverse();

    state_post_ = state_pre_ + kalman_gain * innovation;
    error_cov_post_ =
        (Eigen::MatrixXf::Identity(8, 8) - kalman_gain * measurement_matrix_) *
        error_cov_pre_;

    // 更新可信尺度
    last_reliable_width_ = w;
    last_reliable_height_ = h;
  } else {
    // 使用预测框更新时，仅更新位置，保持尺度不变
    // 1. 保存当前状态
    float old_w = state_post_(2, 0);
    float old_h = state_post_(3, 0);

    // 2. 只更新位置
    Eigen::MatrixXf old_measurement_matrix = measurement_matrix_;
    Eigen::MatrixXf position_only_measurement_matrix =
        Eigen::MatrixXf::Zero(4, 8);
    position_only_measurement_matrix(0, 0) = 1.0f;  // 测量x
    position_only_measurement_matrix(1, 1) = 1.0f;  // 测量y

    measurement_matrix_ = position_only_measurement_matrix;

    // 3. 构造测量值：位置用当前测量，尺度用旧值（会被忽略）
    Eigen::MatrixXf measurement(4, 1);
    measurement(0, 0) = cx;
    measurement(1, 0) = cy;
    measurement(2, 0) = old_w;
    measurement(3, 0) = old_h;

    // 卡尔曼滤波更新步骤
    Eigen::MatrixXf innovation = measurement - measurement_matrix_ * state_pre_;
    Eigen::MatrixXf innovation_covariance =
        measurement_matrix_ * error_cov_pre_ * measurement_matrix_.transpose() +
        measurement_noise_cov_;
    Eigen::MatrixXf kalman_gain = error_cov_pre_ *
                                  measurement_matrix_.transpose() *
                                  innovation_covariance.inverse();

    state_post_ = state_pre_ + kalman_gain * innovation;
    error_cov_post_ =
        (Eigen::MatrixXf::Identity(8, 8) - kalman_gain * measurement_matrix_) *
        error_cov_pre_;

    // 4. 恢复测量矩阵
    measurement_matrix_ = old_measurement_matrix;

    // 5. 保持尺度和尺度速度不变
    state_post_(2, 0) = old_w;  // w
    state_post_(3, 0) = old_h;  // h
    state_post_(6, 0) = 0.0f;   // vw
    state_post_(7, 0) = 0.0f;   // vh
  }

  update_count_++;
}

std::vector<float> KalmanBoxTracker::predict() {
  // 卡尔曼滤波预测步骤
  state_pre_ = transition_matrix_ * state_post_;
  error_cov_pre_ =
      transition_matrix_ * error_cov_post_ * transition_matrix_.transpose() +
      process_noise_cov_;

  // 从状态向量中提取位置和尺度
  float cx = state_pre_(0, 0);
  float cy = state_pre_(1, 0);

  // 获取预测的宽度
  float w_pred = state_pre_(2, 0);

  // 计算最后一次可靠尺度的宽高比
  float aspect_ratio = last_reliable_width_ / last_reliable_height_;

  // 使用预测宽度计算高度
  float w = w_pred;
  float h = w / aspect_ratio;

  float x = cx - w / 2.0f;
  float y = cy - h / 2.0f;

  return {x, y, w, h};
}
