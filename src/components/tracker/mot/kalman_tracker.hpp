#ifndef TRACKER_MOT_KALMAN_TRACKER_HPP
#define TRACKER_MOT_KALMAN_TRACKER_HPP

#include "mot/kalman_filter.hpp"
#include "mot/mot_type_defs.hpp"
#include "tracker/tracker_types.hpp"
class KalmanTracker {
 public:
  KalmanTracker(const uint64_t &frame_id, const KalmanFilter &kf,
                const uint64_t &id, const ObjectBoxInfo &box, int img_width,
                int img_height);
  KalmanTracker() = delete;

  ~KalmanTracker();

 public:
  ObjectBoxInfo box_;
  uint64_t id_;
  KAL_MEAN mean;
  KAL_COVA covariance;

  TrackStatus status_;

  int unmatched_times_;
  int false_update_times_;
  uint64_t ages_;
  int matched_times_;

  bool bounding_;
  float velocity_x_;
  float velocity_y_;

  void predict(const KalmanFilter &kf);
  void update(const uint64_t &frame_id, const KalmanFilter &kf,
              const ObjectBoxInfo *p_bbox, const TrackerConfig &conf);
  void falseUpdateFromPair(const uint64_t &frame_id, const KalmanFilter &kf,
                           KalmanTracker *p_other, const TrackerConfig &conf);
  uint64_t getPairTrackID();

  void updatePairInfo(KalmanTracker *p_other);
  void resetPairInfo();

  ObjectBoxInfo getBoxInfo() const;

 private:
  int img_width_ = 0;
  int img_height_ = 0;
  uint64_t last_updated_frame_id_ = 0;
  std::map<uint64_t, stCorrelateInfo> pair_track_infos_;

  DETECTBOX getBBoxTLWH() const;
  void updateCorre(const DETECTBOX &cur_tlwh, const DETECTBOX &pair_tlwh,
                   stCorrelateInfo &cur_corre, float w_cur);
  void updateBoundaryState();
};

#endif
