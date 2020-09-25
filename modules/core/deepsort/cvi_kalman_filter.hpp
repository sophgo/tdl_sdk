#ifndef _CVI_KALMAN_FILTER_HPP_
#define _CVI_KALMAN_FILTER_HPP_

#include "cvi_kalman_types.hpp"

typedef Eigen::Matrix<float, 1, -1> ROW_VECTOR;

class KalmanFilter {
 public:
  KalmanFilter();
  int predict(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_) const;
  int update(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_, const K_MEASUREMENT_V &z_) const;
  // float mahalanobis(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
  // const K_MEASUREMENT_V &z_) const;
  ROW_VECTOR mahalanobis(const KALMAN_STAGE &s_, const K_STATE_V &x_, const K_COVARIANCE_M &P_,
                         const K_MEASUREMENT_M &Z_) const;

 private:
  K_EXTRAPOLATION_M F_;
  K_OBSERVATION_M H_;
  // K_PROCESS_NOISE_M Q_;
  // K_MEASUREMENT_NOISE_M R_;

  // float std_x_;
  // float std_v_;
};

#endif /* _CVI_KALMAN_FILTER_HPP_ */