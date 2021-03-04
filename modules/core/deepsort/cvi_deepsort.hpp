#ifndef _CVI_DEEPSORT_HPP_
#define _CVI_DEEPSORT_HPP_

#include "cvi_deepsort_types_internal.hpp"
#include "cvi_distance_metric.hpp"
#include "cvi_kalman_filter.hpp"
#include "cvi_kalman_tracker.hpp"
#include "cvi_munkres.hpp"

#include "core/cviai_core.h"

#define USE_COSINE_DISTANCE_FOR_FEATURE true

struct MatchResult {
  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  std::vector<int> unmatched_tracker_idxes;
};

class DeepSORT {
 public:
  DeepSORT();
  DeepSORT(cvai_deepsort_config_t ds_conf);

  static cvai_deepsort_config_t get_DefaultConfig();

  std::vector<std::tuple<bool, uint64_t, TRACKER_STATE, BBOX>> track(
      const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features, int class_id = -1,
      bool use_reid = true);

  int track(cvai_object_t *obj, cvai_tracker_t *tracker_t, bool use_reid = true,
            bool track_by_class = true);
  int track(cvai_face_t *face, cvai_tracker_t *tracker_t, bool use_reid = false);

  void setConfig(cvai_deepsort_config_t ds_conf);

  /* DEBUG CODE */
  void show_INFO_KalmanTrackers();
  std::vector<KalmanTracker> get_Trackers_UnmatchedLastTime() const;
  bool get_Tracker_ByID(uint64_t id, KalmanTracker &tracker) const;
  std::string get_TrackersInfo_UnmatchedLastTime(std::string &str_info) const;

 private:
  uint64_t id_counter;
  std::vector<KalmanTracker> k_trackers;
  KalmanFilter kf_;
  std::vector<int> accreditation_tracker_idxes;
  std::vector<int> probation_tracker_idxes;

  /* deepsort config */
  cvai_deepsort_config_t conf;

  MatchResult match(const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features,
                    const std::vector<int> &Tracker_IDXes, const std::vector<int> &BBox_IDXes,
                    std::string cost_method = "Feature_ConsineDistance",
                    float max_distance = __FLT_MAX__);
  void compute_distance();
  void solve_assignment();
};

#endif /* _CVI_DEEPSORT_HPP_*/
