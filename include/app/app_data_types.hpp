#ifndef APP_DATA_TYPES_HPP
#define APP_DATA_TYPES_HPP
#include "components/snapshot/object_snapshot.hpp"
#include "components/tracker/tracker_types.hpp"
#include "framework/common/common_types.hpp"
#include "framework/image/base_image.hpp"
//--------------------------------
// FaceCapture Data Types
//--------------------------------
struct FaceCaptureResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  std::shared_ptr<BaseImage> image;
  std::vector<ObjectBoxLandmarkInfo> face_boxes;
  std::vector<ObjectBoxInfo> person_boxes;
  std::vector<TrackerInfo> track_results;
  std::vector<ObjectSnapshotInfo> face_snapshots;
};

struct FallDetectionResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  std::shared_ptr<BaseImage> image;
  std::vector<ObjectBoxLandmarkInfo> person_boxes_keypoints;
  std::vector<TrackerInfo> track_results;
  std::map<uint64_t, int> det_results;

};

#endif
