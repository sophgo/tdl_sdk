#ifndef _CVI_DEEPSORT_HPP_
#define _CVI_DEEPSORT_HPP_

#include "cvi_deepsort_types_internal.hpp"
#include "cvi_distance_metric.hpp"
#include "cvi_kalman_filter.hpp"
#include "cvi_kalman_tracker.hpp"
#include "cvi_munkres.hpp"

#include "core/cviai_core.h"

#define MAX_DISTANCE_IOU (float)0.7
#define MAX_DISTANCE_CONSINE (float)0.1

struct MatchResult {
  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  std::vector<int> unmatched_tracker_idxes;
};

class Deepsort {
 public:
  Deepsort();
  Deepsort(int feature_size);

  std::vector<std::tuple<bool, uint64_t, TRACKER_STATE, BBOX>> track(
      const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features);

  int track(cvai_object_t *obj, cvai_tracker_t *tracker_t);

  /* DEBUG CODE */
  void show_INFO_KalmanTrackers();
  std::vector<KalmanTracker> get_Trackers_UnmatchedLastTime() const;
  bool get_Tracker_ByID(uint64_t id, KalmanTracker &tracker) const;

 private:
  uint64_t id_counter;
  // uint32_t feature_size;
  std::vector<KalmanTracker> k_trackers;
  KalmanFilter kf_;
  std::vector<int> accreditation_tracker_idxes;
  std::vector<int> probation_tracker_idxes;

  MatchResult match(const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features,
                    const std::vector<int> &Tracker_IDXes, const std::vector<int> &BBox_IDXes,
                    std::string cost_method = "Feature_ConsineDistance",
                    float max_distance = __FLT_MAX__);
  void compute_distance();
  void solve_assignment();

  static void gateCostMatrix_Mahalanobis(COST_MATRIX &cost_matrix, const KalmanFilter &KF_,
                                         const std::vector<KalmanTracker> &K_Trackers,
                                         const std::vector<BBOX> &BBoxes,
                                         const std::vector<int> &Tracker_IDXes,
                                         const std::vector<int> &BBox_IDXes,
                                         float gate_value = __FLT_MAX__);
};

#endif /* _CVI_DEEPSORT_HPP_*/
