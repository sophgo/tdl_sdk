#include "mot/mot.hpp"
#include "tracker/tracker_types.hpp"
void Tracker::setImgSize(int width, int height) {
  img_width_ = width;
  img_height_ = height;
}

TrackerConfig Tracker::getTrackConfig() { return tracker_config_; }
void Tracker::setTrackConfig(const TrackerConfig &track_config) {
  tracker_config_ = track_config;
}
std::shared_ptr<Tracker> TrackerFactory::createTracker(TrackerType type) {
  switch (type) {
    case TrackerType::TDL_MOT_SORT:
      return std::make_shared<MOT>();
    default:
      return nullptr;
  }
}
