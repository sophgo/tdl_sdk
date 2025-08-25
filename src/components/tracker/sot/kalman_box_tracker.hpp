#ifndef KALMAN_BOX_TRACKER_HPP
#define KALMAN_BOX_TRACKER_HPP

#include <Eigen/Dense>
#include <vector>

/**
 * @brief 卡尔曼滤波器跟踪器类
 *
 * 用于跟踪目标边界框，使用卡尔曼滤波算法预测和更新目标位置和尺寸
 */
class KalmanBoxTracker {
 public:
  /**
   * @brief 构造函数，初始化卡尔曼滤波器
   *
   * @param bbox 初始边界框 [x, y, w, h]
   */
  KalmanBoxTracker(const std::vector<float>& bbox);

  /**
   * @brief 使用观测值更新卡尔曼滤波器状态
   *
   * @param bbox 边界框 [x, y, w, h]
   * @param update_scale 是否更新尺度。使用检测框时为true，使用预测框时为false
   */
  void update(const std::vector<float>& bbox, bool update_scale = true);

  /**
   * @brief 预测下一个状态的边界框位置
   *
   * @return 预测的边界框 [x, y, w, h]
   */
  std::vector<float> predict();

  // 公共成员变量
  int update_count_;  // 更新计数器

 private:
  // 卡尔曼滤波器状态
  Eigen::MatrixXf state_post_;                  // 后验状态估计
  Eigen::MatrixXf error_cov_post_;              // 后验误差协方差
  Eigen::MatrixXf transition_matrix_;           // 状态转移矩阵
  Eigen::MatrixXf measurement_matrix_;          // 测量矩阵
  Eigen::MatrixXf process_noise_cov_;           // 过程噪声协方差
  Eigen::MatrixXf measurement_noise_cov_;       // 测量噪声协方差
  Eigen::MatrixXf base_measurement_noise_cov_;  // 基础测量噪声协方差

  // 记录最后一次可信的尺度
  float last_reliable_width_;
  float last_reliable_height_;

  // 预测状态
  Eigen::MatrixXf state_pre_;
  Eigen::MatrixXf error_cov_pre_;
};

#endif  // KALMAN_BOX_TRACKER_HPP
