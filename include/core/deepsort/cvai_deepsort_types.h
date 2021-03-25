#ifndef _CVI_DEEPSORT_TYPES_H_
#define _CVI_DEEPSORT_TYPES_H_

typedef struct {
  /* for process noise, Q = pow( alpha * x(i) + beta, 2) */
  float Q_std_alpha[8];
  float Q_std_beta[8];
  int Q_std_x_idx[8];
  /* for measurement noise, R = pow( alpha * x(i) + beta, 2) */
  float R_std_alpha[4];
  float R_std_beta[4];
  int R_std_x_idx[4];
} cvai_kalman_filter_config_t;

typedef struct {
  int max_unmatched_num;
  int accreditation_threshold;
  int feature_budget_size;
  int feature_update_interval;
  /* for initial covariance, P = pow( alpha * x(i) + beta, 2) */
  float P_std_alpha[8];
  float P_std_beta[8];
  int P_std_x_idx[8];
} cvai_kalman_tracker_config_t;

typedef struct {
  float max_distance_iou;
  float max_distance_consine;
  int max_unmatched_times_for_bbox_matching;
  cvai_kalman_filter_config_t kfilter_conf;
  cvai_kalman_tracker_config_t ktracker_conf;
} cvai_deepsort_config_t;

#endif /* _CVI_DEEPSORT_TYPES_H_ */
