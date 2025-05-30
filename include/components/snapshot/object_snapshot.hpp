#ifndef TDL_SDK_OBJECT_CAPTURE_HPP
#define TDL_SDK_OBJECT_CAPTURE_HPP

#include <json.hpp>
#include <memory>
#include "common/model_output_types.hpp"
#include "components/tracker/tracker_types.hpp"
#include "framework/common/packet.hpp"
#include "framework/image/base_image.hpp"
#include "framework/preprocess/base_preprocessor.hpp"
struct ObjectSnapshotInfo {
  float quality = 0;
  std::shared_ptr<BaseImage> object_image;
  uint64_t snapshot_frame_id = 0;
  uint64_t export_frame_id = 0;
  uint64_t track_id = 0;
  int32_t miss_counter = 0;
  ObjectBoxInfo object_box_info;  // this is the box of the cropped image
  std::map<std::string, Packet> other_info;
};

struct SnapshotConfig {
  int max_miss_counter;
  int snapshot_interval;  // if 0,only export once
  int min_snapshot_size;  // if 0,no limit
  int crop_size_min;
  int crop_size_max;
  float snapshot_quality_threshold;  // if 0,no limit,range
  float update_quality_gap;
  bool crop_square;
};

// 基类
class ObjectSnapshot {
 public:
  ObjectSnapshot();
  virtual ~ObjectSnapshot() = default;
  void setConfig(SnapshotConfig config) { config_ = config; }
  SnapshotConfig getConfig() { return config_; }
  int32_t updateConfig(const nlohmann::json& config);
  int32_t updateSnapshot(std::shared_ptr<BaseImage> image, uint64_t frame_id,
                         const std::map<uint64_t, ObjectBoxInfo>& track_boxes,
                         const std::vector<TrackerInfo>& tracks,
                         const std::map<uint64_t, float>& quality_scores,
                         const std::map<std::string, Packet>& other_info,
                         const std::map<uint64_t, std::shared_ptr<BaseImage>>&
                             crop_face_imgs = {});

  int32_t getSnapshotData(std::vector<ObjectSnapshotInfo>& snapshots,
                          bool force_all = false);

 private:
  int32_t getCropBox(ObjectBoxInfo& box, int& x, int& y, int& width,
                     int& height, int& dst_width, int& dst_height,
                     int img_width, int img_height);

  void resetSnapshotInfo(ObjectSnapshotInfo& info, uint64_t frame_id);

 protected:
  SnapshotConfig config_;
  std::map<uint64_t, ObjectSnapshotInfo> snapshot_infos_;
  std::vector<ObjectSnapshotInfo> export_snapshots_;
  std::shared_ptr<BasePreprocessor> preprocessor_;
};

#endif /* TDL_SDK_OBJECT_CAPTURE_HPP */