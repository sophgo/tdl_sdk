#ifndef TDL_SDK_TRACKER_TYPES_HPP
#define TDL_SDK_TRACKER_TYPES_HPP

#include <memory>
#include "common/model_output_types.hpp"
#include "common/object_type_def.hpp"
#include "image/base_image.hpp"
#include "model/base_model.hpp"

enum class TrackStatus { NEW = 0, TRACKED, LOST, REMOVED };
class TrackerInfo {
 public:
  TrackerInfo() = default;
  ~TrackerInfo() = default;

  ObjectBoxInfo box_info_;
  TrackStatus status_;
  int matched_times_;
  // if -1, it is not matched with current detection
  int obj_idx_;
  int pair_track_idx_;
  uint64_t track_id_;
  float velocity_x_;
  float velocity_y_;
  float blurness;
};
class TrackerConfig {
 public:
  int max_unmatched_times_ = 15;
  int track_confirmed_frames_ = 3;
  int track_pair_update_missed_times_ = 2;
  float track_init_score_thresh_ = 0.6;
  float high_score_thresh_ = 0.5;
  float high_score_iou_dist_thresh_ = 0.7;
  float low_score_iou_dist_thresh_ = 0.5;
};

enum class TrackerType {
  TDL_MOT_SORT = 0,
  TDL_SOT = 1,
};
class Tracker {
 public:
  Tracker() = default;
  ~Tracker() = default;

  virtual int32_t setModel(std::shared_ptr<BaseModel> sot_model);

  virtual int32_t initialize(const std::shared_ptr<BaseImage>& image,
                             const std::vector<ObjectBoxInfo>& detect_boxes,
                             const ObjectBoxInfo& bbox, int frame_type);

  virtual int32_t initialize(const std::shared_ptr<BaseImage>& image,
                             const std::vector<ObjectBoxInfo>& detect_boxes,
                             float x, float y, int frame_type);

  virtual int32_t initialize(const std::shared_ptr<BaseImage>& image,
                             const std::vector<ObjectBoxInfo>& detect_boxes,
                             int index);

  virtual void setPairConfig(
      std::map<TDLObjectType, TDLObjectType> object_pair_config){};

  void setTrackConfig(const TrackerConfig& track_config);

  TrackerConfig getTrackConfig();

  virtual int32_t track(std::vector<ObjectBoxInfo>& boxes, uint64_t frame_id,
                        std::vector<TrackerInfo>& trackers);

  virtual int32_t track(const std::shared_ptr<BaseImage>& image,
                        uint64_t frame_id, TrackerInfo& tracker_info);

  void setImgSize(int width, int height);

 protected:
  int img_width_ = 0;
  int img_height_ = 0;
  TrackerConfig tracker_config_;
};

class TrackerFactory {
 public:
  static std::shared_ptr<Tracker> createTracker(TrackerType type);
};

#endif /* TDL_SDK_TRACKER_TYPES_HPP */
