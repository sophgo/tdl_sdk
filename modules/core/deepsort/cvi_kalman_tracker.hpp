#ifndef _CVI_KALMAN_TRACKER_HPP_
#define _CVI_KALMAN_TRACKER_HPP_

#include <vector>
#include "cvi_deepsort_types_internal.hpp"
#include "cvi_distance_metric.hpp"
#include "cvi_kalman_filter.hpp"
#include "cvi_kalman_types.hpp"
#include "cvi_tracker.hpp"

#define USE_COSINE_DISTANCE_FOR_FEATURE true

#define MAX_UNMATCHED_TIMES_FOR_BBOX_MATCHING 2

#define MAX_UNMATCHED_NUM 40
#define ACCREDITATION_THRESHOLD 3
#define FEATURE_BUDGET_SIZE 8
#define FEATURE_UPDATE_INTERVAL 1

// typedef COST_MATRIX (*CostMarixFunction_KTracker)(const
// std::vector<KalmanTracker> &KTrackers,
//                     const std::vector<BBOX> &BBoxes,
//                     const std::vector<FEATURE> &Features,
//                     const std::vector<int> &Tracker_IDs,
//                     const std::vector<int> &BBox_IDs);

enum TRACKER_STATE { MISS = 0, PROBATION, ACCREDITATION };

class KalmanTracker : public Tracker {
 public:
  std::vector<FEATURE> features_;
  KALMAN_STAGE kalman_state_;
  TRACKER_STATE tracker_state_;
  K_STATE_V x_;
  K_COVARIANCE_M P_;
  int unmatched_times;
  KalmanTracker();
  KalmanTracker(const uint64_t &id, const BBOX &bbox, const FEATURE &feature);
  KalmanTracker(const uint64_t &id, const BBOX &bbox, const FEATURE &feature,
                const cvai_kalman_tracker_config_t &ktracker_conf);

  void update_state(bool is_matched);
  void update_bbox(const BBOX &bbox);
  void update_feature(const FEATURE &feature);

  BBOX getBBox_TLWH() const;

  static COST_MATRIX getCostMatrix_Feature(const std::vector<KalmanTracker> &KTrackers,
                                           const std::vector<BBOX> &BBoxes,
                                           const std::vector<FEATURE> &Features,
                                           const std::vector<int> &Tracker_IDs,
                                           const std::vector<int> &BBox_IDs);

  static COST_MATRIX getCostMatrix_BBox(const std::vector<KalmanTracker> &KTrackers,
                                        const std::vector<BBOX> &BBoxes,
                                        const std::vector<FEATURE> &Features,
                                        const std::vector<int> &Tracker_IDs,
                                        const std::vector<int> &BBox_IDs);
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