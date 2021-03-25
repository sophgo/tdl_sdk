#ifndef _CVI_KALMAN_FILTER_HPP_
#define _CVI_KALMAN_FILTER_HPP_

#include "core/deepsort/cvai_deepsort_types.h"
#include "cvi_kalman_types.hpp"

typedef Eigen::Matrix<float, 1, -1> ROW_VECTOR;

class KalmanFilter {
 public:
  KalmanFilter();
  int predict(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_) const;
  int update(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_, const K_MEASUREMENT_V &z_) const;
  ROW_VECTOR mahalanobis(const KALMAN_STAGE &s_, const K_STATE_V &x_, const K_COVARIANCE_M &P_,
                         const K_MEASUREMENT_M &Z_) const;
  int predict(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
              const cvai_kalman_filter_config_t &kfilter_conf) const;
  int update(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_, const K_MEASUREMENT_V &z_,
             const cvai_kalman_filter_config_t &kfilter_conf) const;
  ROW_VECTOR mahalanobis(const KALMAN_STAGE &s_, const K_STATE_V &x_, const K_COVARIANCE_M &P_,
                         const K_MEASUREMENT_M &Z_,
                         const cvai_kalman_filter_config_t &kfilter_conf) const;
#if 0
  float mahalanobis(KALMAN_STAGE &s_, K_STATE_V &x_, K_COVARIANCE_M &P_,
                    const K_MEASUREMENT_V &z_) const;
#endif

 private:
  K_EXTRAPOLATION_M F_;
  K_OBSERVATION_M H_;
};

#endif /* _CVI_KALMAN_FILTER_HPP_ */