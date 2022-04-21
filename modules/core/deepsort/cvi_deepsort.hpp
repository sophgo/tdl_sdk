#ifndef _CVI_DEEPSORT_HPP_
#define _CVI_DEEPSORT_HPP_

#include "cvi_deepsort_types_internal.hpp"
#include "cvi_distance_metric.hpp"
#include "cvi_kalman_filter.hpp"
#include "cvi_kalman_tracker.hpp"
#include "cvi_munkres.hpp"

#include "core/cviai_core.h"

struct MatchResult {
  std::vector<std::pair<int, int>> matched_pairs;
  std::vector<int> unmatched_bbox_idxes;
  std::vector<int> unmatched_tracker_idxes;
};

/* Result Format: [i] <is_matched, tracker_id, tracker_state, tracker_bbox> */
typedef std::vector<std::tuple<bool, uint64_t, k_tracker_state_e, BBOX>> Tracking_Result;

class DeepSORT {
 public:
  DeepSORT() = delete;
  DeepSORT(bool use_specific_counter);
  ~DeepSORT();

  static cvai_deepsort_config_t get_DefaultConfig();

  CVI_S32 track(Tracking_Result &result, const std::vector<BBOX> &BBoxes,
                const std::vector<FEATURE> &Features, int class_id = -1, bool use_reid = true,
                float *Quality = NULL);

  CVI_S32 track(cvai_object_t *obj, cvai_tracker_t *tracker, bool use_reid = true);
  CVI_S32 track(cvai_face_t *face, cvai_tracker_t *tracker, bool use_reid = false);

  CVI_S32 getConfig(cvai_deepsort_config_t *ds_conf, int cviai_obj_type = -1);
  CVI_S32 setConfig(cvai_deepsort_config_t *ds_conf, int cviai_obj_type = -1,
                    bool show_config = false);
  void cleanCounter();

  CVI_S32 get_trackers_inactive(cvai_tracker_t *tracker) const;

  /* DEBUG CODE */
  // TODO: refactor these functions.
  void show_INFO_KalmanTrackers();
  std::vector<KalmanTracker> get_Trackers_UnmatchedLastTime() const;
  bool get_Tracker_ByID(uint64_t id, KalmanTracker &tracker) const;
  std::string get_TrackersInfo_UnmatchedLastTime(std::string &str_info) const;

 private:
  bool sp_counter;
  uint64_t id_counter;
  std::map<int, uint64_t> specific_id_counter;
  std::vector<KalmanTracker> k_trackers;
  KalmanFilter kf_;
  std::vector<int> accreditation_tracker_idxes;
  std::vector<int> probation_tracker_idxes;

  /* DeepSORT config */
  cvai_deepsort_config_t default_conf;
  std::map<int, cvai_deepsort_config_t> specific_conf;

  uint64_t get_nextID(int class_id);

  MatchResult match(const std::vector<BBOX> &BBoxes, const std::vector<FEATURE> &Features,
                    const std::vector<int> &Tracker_IDXes, const std::vector<int> &BBox_IDXes,
                    cvai_kalman_filter_config_t &kf_conf,
                    cost_matrix_algo_e cost_method = Feature_CosineDistance,
                    float max_distance = __FLT_MAX__);
  void compute_distance();
  void solve_assignment();
};

#endif /* _CVI_DEEPSORT_HPP_*/
