#ifndef TRACKER_COST_MATRIX_HELPER_HPP
#define TRACKER_COST_MATRIX_HELPER_HPP

#include "mot/kalman_filter.hpp"
#include "mot/kalman_tracker.hpp"
#include "mot/mot_type_defs.hpp"

class CostMatrixHelper {
 public:
  CostMatrixHelper();
  static COST_MATRIX getCostMatrixFeature(
      const std::vector<std::shared_ptr<KalmanTracker>> &trackers,
      const std::vector<ObjectBoxInfo> &detections,
      const std::vector<ModelFeatureInfo> &features,
      const std::vector<int> &tracker_idx,
      const std::vector<int> &detection_idx);

  static COST_MATRIX getCostMatrixBBox(
      const std::vector<std::shared_ptr<KalmanTracker>> &trackers,
      const std::vector<ObjectBoxInfo> &detections,
      const std::vector<int> &tracker_idx,
      const std::vector<int> &detection_idx);

  //   static void restrictCostMatrixMahalanobis(
  //       COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
  //       const std::vector<KalmanTracker> &trackers,
  //       const std::vector<ObjectBoxInfo> &detections,
  //       const std::vector<int> &tracker_idx,
  //       const std::vector<int> &detection_idx,
  //       const mot_kalman_filter_config_t &kfilter_conf, float upper_bound);
};

#endif
