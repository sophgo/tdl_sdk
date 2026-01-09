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

struct FacePetCaptureResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  std::shared_ptr<BaseImage> image;
  std::vector<ObjectBoxLandmarkInfo> face_boxes;
  std::vector<ObjectBoxInfo> person_boxes;
  std::vector<ObjectBoxInfo> pet_boxes;
  std::vector<TrackerInfo> track_results;
  std::vector<ObjectSnapshotInfo> face_snapshots;
  std::map<uint64_t, std::vector<float>> face_features;
  std::vector<std::map<TDLObjectAttributeType, float>> face_attributes;
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

struct ConsumerCountingResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t enter_num;
  uint32_t miss_num;
  std::shared_ptr<BaseImage> image;
  std::vector<ObjectBoxInfo> object_boxes;
  std::vector<TrackerInfo> track_results;
  std::vector<uint64_t> cross_id;
  std::vector<int> counting_line;
};

struct HumanPoseResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  std::shared_ptr<BaseImage> image;
  std::vector<ObjectBoxLandmarkInfo> person_boxes_keypoints;
  std::vector<TrackerInfo> track_results;
};

#endif
