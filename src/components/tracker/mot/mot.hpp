#ifndef TRACKER_MOT_MOT_HPP
#define TRACKER_MOT_MOT_HPP

#include <map>
#include <vector>
#include "mot/kalman_filter.hpp"
#include "mot/kalman_tracker.hpp"
#include "mot/mot_type_defs.hpp"
#include "tracker/tracker_types.hpp"
class MOT : public Tracker {
 public:
  MOT();
  ~MOT();

  int32_t track(std::vector<ObjectBoxInfo> &boxes, uint64_t frame_id,
                std::vector<TrackerInfo> &trackers);
  void setPairConfig(
      std::map<TDLObjectType, TDLObjectType> object_pair_config) {
    object_pair_config_ = object_pair_config;
  }

 private:
  void trackAlone(std::vector<ObjectBoxInfo> &boxes, TDLObjectType obj_type);
  void trackFuse(std::vector<ObjectBoxInfo> &boxes, TDLObjectType priority_type,
                 TDLObjectType secondary_type);
  void updatePairInfo(std::vector<ObjectBoxInfo> &boxes,
                      const std::vector<int> &priority_idxes,
                      const std::vector<int> &secondary_idxes,
                      TDLObjectType priority_type, TDLObjectType secondary_type,
                      float corre_thresh);

  MatchResult match(const std::vector<ObjectBoxInfo> &dets,
                    const std::vector<ModelFeatureInfo> &features,
                    const std::vector<int> &tracker_idxes,
                    const std::vector<int> &det_idxes,
                    TrackCostType cost_method = TrackCostType::BBOX_IOU,
                    float max_distance = __FLT_MAX__);

  void updateTrackers(const std::vector<ObjectBoxInfo> &boxes,
                      const std::vector<ModelFeatureInfo> &features);
  void resetPairTrackerOfRemovedTracker(uint64_t tracker_id);

  std::map<uint64_t, uint64_t> getPairTrackIds();
  KalmanFilter kalman_filter_;
  std::vector<std::shared_ptr<KalmanTracker>> trackers_;
  std::vector<int> pair_obj_idxes_;
  std::vector<uint64_t> det_track_ids_;

  // the key is the priority object, will be tracked first
  std::map<TDLObjectType, TDLObjectType> object_pair_config_;

  float conf_thresh_high_ = 0.45;  //
  float conf_thresh_low_ = 0.3;
  uint64_t id_counter_ = 0;
  uint64_t current_frame_id_ = 0;
};

#endif /* TRACKER_MOT_MOT_HPP */
