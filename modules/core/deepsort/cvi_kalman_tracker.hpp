#ifndef _CVI_KALMAN_TRACKER_HPP_
#define _CVI_KALMAN_TRACKER_HPP_

#include <vector>
#include "cvi_deepsort_types_internal.hpp"
#include "cvi_distance_metric.hpp"
#include "cvi_kalman_filter.hpp"
#include "cvi_kalman_types.hpp"
#include "cvi_tracker.hpp"

enum TRACKER_STATE { MISS = 0, PROBATION, ACCREDITATION };

const float chi2inv95[10] = {0,      3.8415, 5.9915, 7.8147, 9.4877,
                             11.070, 12.592, 14.067, 15.507, 16.919};

class KalmanTracker : public Tracker {
 public:
  std::vector<FEATURE> features_;
  KALMAN_STAGE kalman_state_;
  TRACKER_STATE tracker_state_;
  K_STATE_V x_;
  K_COVARIANCE_M P_;
  int unmatched_times;
  KalmanTracker();
  KalmanTracker(const uint64_t &id, const int &class_id, const BBOX &bbox, const FEATURE &feature);
  KalmanTracker(const uint64_t &id, const int &class_id, const BBOX &bbox, const FEATURE &feature,
                const cvai_kalman_tracker_config_t &ktracker_conf);

  void update_state(bool is_matched, int max_unmatched_num = 40, int accreditation_thr = 3);
  void update_bbox(const BBOX &bbox);
  void update_feature(const FEATURE &feature, int feature_budget_size = 8,
                      int feature_update_interval = 1);

  BBOX getBBox_TLWH() const;

  static COST_MATRIX getCostMatrix_Feature(const std::vector<KalmanTracker> &KTrackers,
                                           const std::vector<BBOX> &BBoxes,
                                           const std::vector<FEATURE> &Features,
                                           const std::vector<int> &Tracker_IDXes,
                                           const std::vector<int> &BBox_IDXes);

  static COST_MATRIX getCostMatrix_BBox(const std::vector<KalmanTracker> &KTrackers,
                                        const std::vector<BBOX> &BBoxes,
                                        const std::vector<FEATURE> &Features,
                                        const std::vector<int> &Tracker_IDXes,
                                        const std::vector<int> &BBox_IDXes);

  static COST_MATRIX getCostMatrix_Mahalanobis(const KalmanFilter &KF_,
                                               const std::vector<KalmanTracker> &K_Trackers,
                                               const std::vector<BBOX> &BBoxes,
                                               const std::vector<int> &Tracker_IDXes,
                                               const std::vector<int> &BBox_IDXes,
                                               const cvai_kalman_filter_config_t &kfilter_conf,
                                               float gate_value = __FLT_MAX__);

  static void gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                         const std::vector<KalmanTracker> &K_Trackers,
                                         const std::vector<BBOX> &BBoxes,
                                         const std::vector<int> &Tracker_IDXes,
                                         const std::vector<int> &BBox_IDXes,
                                         float gate_value = __FLT_MAX__);

  static void gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                         const std::vector<KalmanTracker> &K_Trackers,
                                         const std::vector<BBOX> &BBoxes,
                                         const std::vector<int> &Tracker_IDXes,
                                         const std::vector<int> &BBox_IDXes,
                                         const cvai_kalman_filter_config_t &kfilter_conf,
                                         float gate_value = __FLT_MAX__);

  /* DEBUG CODE */
  int get_FeatureUpdateCounter() const;
  int get_MatchedCounter() const;
  std::string get_INFO_KalmanState() const;
  std::string get_INFO_TrackerState() const;

 private:
  int feature_update_counter;
  int matched_counter;
};

#endif /* _CVI_KALMAN_TRACKER_HPP_ */