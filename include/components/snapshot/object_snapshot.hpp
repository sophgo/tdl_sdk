#ifndef TDL_SDK_OBJECT_CAPTURE_HPP
#define TDL_SDK_OBJECT_CAPTURE_HPP

#include <json.hpp>
#include <memory>
#include <unordered_set>
#include "common/model_output_types.hpp"
#include "components/tracker/tracker_types.hpp"
#include "encoder/image_encoder/image_encoder.hpp"
#include "framework/common/packet.hpp"
#include "framework/image/base_image.hpp"
#include "framework/preprocess/base_preprocessor.hpp"
#include "nn/tdl_model_factory.hpp"

struct ObjectSnapshotInfo {
  float quality = 0;
  std::shared_ptr<BaseImage> object_image;
  std::vector<uint8_t> encoded_full_image;
  uint64_t snapshot_frame_id = 0;
  uint64_t export_frame_id = 0;
  uint64_t track_id = 0;
  uint64_t pair_track_id = 0;
  int32_t registered_id =
      -2;  // -2:已由人脸进行确认；-1：陌生人；其余对应注册用户ID
  int32_t miss_counter = 0;
  int32_t matched_times = 0;
  ObjectBoxInfo object_box_info;  // this is the box of the cropped image
  std::map<std::string, Packet> other_info;
};

struct SnapshotConfig {
  int max_miss_counter;
  int min_matched_times;
  int snapshot_interval;  // if 0,only export once
  int min_snapshot_size;  // if 0,no limit
  int crop_size_min;
  int crop_size_max;
  float snapshot_quality_threshold;         // if 0,no limit,range
  float snapshot_person_quality_threshold;  // if 0,no limit,range
  float update_quality_gap;
  bool crop_square;
  int jpg_quality;
  bool use_reid;
  float reid_threshold;
};

// 基类
class ObjectSnapshot {
 public:
  ObjectSnapshot(int vechn = -1, int encoder_mode = -1);
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
  std::shared_ptr<ImageEncoder> getImageEncoder() { return image_encoder_; }

  bool isCapturePerson() { return capture_person_; }
  void setReidModel(std::shared_ptr<BaseModel> model) { reid_model_ = model; }
  void setFaceID(uint64_t person_track_id, int registered_id);

 private:
  int32_t getCropBox(ObjectBoxInfo& box, int& x, int& y, int& width,
                     int& height, int& dst_width, int& dst_height,
                     int img_width, int img_height);

  void resetSnapshotInfo(ObjectSnapshotInfo& info, uint64_t frame_id);
  std::shared_ptr<ImageEncoder> image_encoder_;
  std::shared_ptr<BaseModel> reid_model_;

 protected:
  bool capture_person_ = false;
  SnapshotConfig config_;
  std::map<uint64_t, ObjectSnapshotInfo> snapshot_infos_;
  std::vector<ObjectSnapshotInfo> export_snapshots_;
  std::shared_ptr<BasePreprocessor> preprocessor_;

  // ReID assisted confirmation
  std::map<uint64_t, std::vector<float>>
      reid_feature_map_;                        // Initial ID -> Feature
  std::map<uint64_t, uint64_t> person_id_map_;  // New ID -> Initial ID
  std::map<uint64_t, uint64_t>
      person_faceID_map_;  // Initial ID -> Registered Face ID
  std::unordered_set<uint64_t> valid_face_ids_;
};

#endif /* TDL_SDK_OBJECT_CAPTURE_HPP */