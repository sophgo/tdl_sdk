#include "cvi_kalman_filter.hpp"
#include <math.h>

#include <iostream>
#include "cviai_log.hpp"

#define DEBUG_KALMAN_FILTER_UPDATE 0
#define DEBUG_KALMAN_FILTER_MAHALANOBIS 0

KalmanFilter::KalmanFilter() {
  LOGD("Create Kalman Filter");
  F_ = Eigen::MatrixXf::Identity(DIM_X, DIM_X);
  F_.topRightCorner(DIM_Z, DIM_Z) = Eigen::MatrixXf::Identity(DIM_Z, DIM_Z);
  H_ = Eigen::MatrixXf::Identity(DIM_Z, DIM_X);
}

int KalmanFilter::predict(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_) const {
  assert(s_ == KALMAN_STAGE::UPDATED);
  /* generate process noise, Q */
  K_PROCESS_NOISE_M Q_ = Eigen::MatrixXf::Zero(DIM_X, DIM_X);
  for (int i = 0; i < DIM_Z; i++) {
    Q_(i, i) = pow(STD_XP_0 * x_(3), 2);
  }
  for (int i = DIM_Z; i < DIM_X; i++) {
    Q_(i, i) = pow(STD_XP_1 * x_(3), 2);
  }
  Q_(DIM_Z - 2, DIM_Z - 2) = pow(1e-2, 2);
  Q_(DIM_X - 2, DIM_X - 2) = pow(1e-5, 2);

  /* predict */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

  s_ = KALMAN_STAGE::PREDICTED;
  return 0;
}

int KalmanFilter::predict(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
                          const cvai_kalman_filter_config_t &kfilter_conf) const {
  assert(s_ == KALMAN_STAGE::UPDATED);
  /* generate process noise, Q */
  K_PROCESS_NOISE_M Q_ = Eigen::MatrixXf::Zero(DIM_X, DIM_X);
  for (int i = 0; i < DIM_X; i++) {
    if (kfilter_conf.Q_std_x_idx[i] != -1) {
      Q_(i, i) = pow(kfilter_conf.Q_std_alpha[i] * x_[kfilter_conf.Q_std_x_idx[i]] +
                         kfilter_conf.Q_std_beta[i],
                     2);
    } else {
      Q_(i, i) = pow(kfilter_conf.Q_std_beta[i], 2);
    }
  }
  /* predict */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;

  s_ = KALMAN_STAGE::PREDICTED;
  return 0;
}

int KalmanFilter::update(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
                         const K_MEASUREMENT_V &z_) const {
  assert(s_ == KALMAN_STAGE::PREDICTED);
  /* Compute the Kalman Gain : K = P * H^t * (H * P * H^t + R)^(-1) */
  /* generate measurement noise, R */
  K_MATRIX_Z_Z R_ = Eigen::MatrixXf::Zero(DIM_Z, DIM_Z);
  for (int i = 0; i < DIM_Z; i++) {
    R_(i, i) = pow(STD_XP_0 * x_(3), 2);
  }
  R_(DIM_Z - 2, DIM_Z - 2) = pow(1e-1, 2);
  // K_MATRIX_Z_Z HPHt_ = H_ * P_ * H_.transpose();
  K_MATRIX_Z_Z HPHt_ = P_.block(0, 0, DIM_Z, DIM_Z);
  HPHt_ = HPHt_ + R_;

  // K_MATRIX_X_Z PHt_ = P_ * H_.transpose();
  K_MATRIX_X_Z PHt_ = P_.block(0, 0, DIM_X, DIM_Z);

  // K_MATRIX_X_Z K_ = PHt_ * HPHt_.inverse();
  K_MATRIX_X_Z K_ = HPHt_.llt().solve(PHt_.transpose()).transpose();

  /* Update estimate with measurement : x = x + K * (z - H * x) */
  // K_MEASUREMENT_V Hx_ = H_ * x_;
  K_MEASUREMENT_V Hx_ = x_.block(0, 0, DIM_Z, 1);
  x_ = x_ + K_ * (z_ - Hx_);

  /* Update the estimate uncertainty : P = (I - K * H) * P */
  P_ = P_ - K_ * P_.block(0, 0, DIM_Z, DIM_X).matrix();

  s_ = KALMAN_STAGE::UPDATED;
  return 0;
}

int KalmanFilter::update(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
                         const K_MEASUREMENT_V &z_,
                         const cvai_kalman_filter_config_t &kfilter_conf) const {
  assert(s_ == KALMAN_STAGE::PREDICTED);
  /* Compute the Kalman Gain : K = P * H^t * (H * P * H^t + R)^(-1) */
  /* generate measurement noise, R */
  K_MATRIX_Z_Z R_ = Eigen::MatrixXf::Zero(DIM_Z, DIM_Z);
  for (int i = 0; i < DIM_Z; i++) {
    if (kfilter_conf.R_std_x_idx[i] != -1) {
      R_(i, i) = pow(kfilter_conf.R_std_alpha[i] * x_[kfilter_conf.R_std_x_idx[i]] +
                         kfilter_conf.R_std_beta[i],
                     2);
    } else {
      R_(i, i) = pow(kfilter_conf.R_std_beta[i], 2);
    }
  }
  // K_MATRIX_Z_Z HPHt_ = H_ * P_ * H_.transpose();
  K_MATRIX_Z_Z HPHt_ = P_.block(0, 0, DIM_Z, DIM_Z);
  HPHt_ = HPHt_ + R_;

  // K_MATRIX_X_Z PHt_ = P_ * H_.transpose();
  K_MATRIX_X_Z PHt_ = P_.block(0, 0, DIM_X, DIM_Z);

  // K_MATRIX_X_Z K_ = PHt_ * HPHt_.inverse();
  K_MATRIX_X_Z K_ = HPHt_.llt().solve(PHt_.transpose()).transpose();

  /* Update estimate with measurement : x = x + K * (z - H * x) */
  // K_MEASUREMENT_V Hx_ = H_ * x_;
  K_MEASUREMENT_V Hx_ = x_.block(0, 0, DIM_Z, 1);
  x_ = x_ + K_ * (z_ - Hx_);

  /* Update the estimate uncertainty : P = (I - K * H) * P */
  P_ = P_ - K_ * P_.block(0, 0, DIM_Z, DIM_X).matrix();

  s_ = KALMAN_STAGE::UPDATED;
  return 0;
}

ROW_VECTOR KalmanFilter::mahalanobis(const KALMAN_STAGE &s_, const K_STATE_V &x_,
                                     const K_COVARIANCE_M &P_, const K_MEASUREMENT_M &Z_) const {
  K_MATRIX_Z_Z Cov_ = P_.block(0, 0, DIM_Z, DIM_Z);
  K_MATRIX_Z_Z R_ = Eigen::MatrixXf::Zero(DIM_Z, DIM_Z);
  for (int i = 0; i < DIM_Z; i++) {
    R_(i, i) = pow(STD_XP_0 * x_(3), 2);
  }
  R_(DIM_Z - 1, DIM_Z - 1) = pow(1e-1, 2);
  Cov_ += R_;

  K_MEASUREMENT_V Hx_ = x_.block(0, 0, DIM_Z, 1);

  K_MEASUREMENT_M Diff_ = Z_.rowwise() - Hx_.transpose();

  /*
    d = x^t * S^(-1) * x
      = x^t * (L * L^t)^(-1) * x
      = x^t * (L^t)^(-1) * L^(-1) * x
      = (L^(-1) * x)^t * (L^(-1) * x)
      = M^t * M
  */
  Eigen::Matrix<float, -1, -1> L_ = Cov_.llt().matrixL();
  Eigen::Matrix<float, -1, -1> M_ =
      L_.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(Diff_).transpose();

  auto M_2_ = ((M_.array()) * (M_.array())).matrix();
  auto maha2_distance = M_2_.colwise().sum();

  return maha2_distance;
}

ROW_VECTOR KalmanFilter::mahalanobis(const KALMAN_STAGE &s_, const K_STATE_V &x_,
                                     const K_COVARIANCE_M &P_, const K_MEASUREMENT_M &Z_,
                                     const cvai_kalman_filter_config_t &kfilter_conf) const {
  K_MATRIX_Z_Z Cov_ = P_.block(0, 0, DIM_Z, DIM_Z);
  K_MATRIX_Z_Z R_ = Eigen::MatrixXf::Zero(DIM_Z, DIM_Z);
  for (int i = 0; i < DIM_Z; i++) {
    if (kfilter_conf.R_std_x_idx[i] != -1) {
      R_(i, i) = pow(kfilter_conf.R_std_alpha[i] * x_[kfilter_conf.R_std_x_idx[i]] +
                         kfilter_conf.R_std_beta[i],
                     2);
    } else {
      R_(i, i) = pow(kfilter_conf.R_std_beta[i], 2);
    }
  }
  Cov_ += R_;

  K_MEASUREMENT_V Hx_ = x_.block(0, 0, DIM_Z, 1);

  K_MEASUREMENT_M Diff_ = Z_.rowwise() - Hx_.transpose();

  /*
    d = x^t * S^(-1) * x
      = x^t * (L * L^t)^(-1) * x
      = x^t * (L^t)^(-1) * L^(-1) * x
      = (L^(-1) * x)^t * (L^(-1) * x)
      = M^t * M
  */
  Eigen::Matrix<float, -1, -1> L_ = Cov_.llt().matrixL();
  Eigen::Matrix<float, -1, -1> M_ =
      L_.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(Diff_).transpose();

  auto M_2_ = ((M_.array()) * (M_.array())).matrix();
  auto maha2_distance = M_2_.colwise().sum();

  return maha2_distance;
}