#include "mot/mot.hpp"
#include "sot/sot.hpp"
#include "tracker/tracker_types.hpp"
void Tracker::setImgSize(int width, int height) {
  img_width_ = width;
  img_height_ = height;
}

int32_t Tracker::setModel(std::shared_ptr<BaseModel> sot_model) { return 0; }

int32_t Tracker::initialize(const std::shared_ptr<BaseImage>& image,
                            const std::vector<ObjectBoxInfo>& detect_boxes,
                            const ObjectBoxInfo& bbox) {
  return 0;
}

int32_t Tracker::initialize(const std::shared_ptr<BaseImage>& image,
                            const std::vector<ObjectBoxInfo>& detect_boxes,
                            float x, float y) {
  return 0;
}

TrackerConfig Tracker::getTrackConfig() { return tracker_config_; }

void Tracker::setTrackConfig(const TrackerConfig& track_config) {
  tracker_config_ = track_config;
}

int32_t Tracker::track(std::vector<ObjectBoxInfo>& boxes, uint64_t frame_id,
                       std::vector<TrackerInfo>& trackers) {
  return 0;
}

int32_t Tracker::track(const std::shared_ptr<BaseImage>& image,
                       uint64_t frame_id, TrackerInfo& tracker_info) {
  return 0;
}

std::shared_ptr<Tracker> TrackerFactory::createTracker(TrackerType type) {
  switch (type) {
    case TrackerType::TDL_MOT_SORT:
      return std::make_shared<MOT>();
    case TrackerType::TDL_SOT:
      return std::make_shared<SOT>();
    default:
      return nullptr;
  }
}