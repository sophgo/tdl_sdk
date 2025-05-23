#include "components/snapshot/object_snapshot.hpp"
#include "framework/utils/tdl_log.hpp"
ObjectSnapshot::ObjectSnapshot() {
  preprocessor_ =
      PreprocessorFactory::createPreprocessor(InferencePlatform::AUTOMATIC);
  if (preprocessor_ == nullptr) {
    LOGE("ObjectSnapshot preprocessor is nullptr");
    assert(false);
  }
  config_.update_quality_gap = 0.05f;
  config_.snapshot_quality_threshold = 0.f;
  config_.crop_square = false;
  config_.max_miss_counter = 20;
  config_.snapshot_interval = 10;
  config_.crop_size_min = 0;
  config_.crop_size_max = 0;
  config_.min_snapshot_size = 0;
}

int32_t ObjectSnapshot::updateConfig(const nlohmann::json& config) {
  LOGI("ObjectSnapshot update config, config: %s", config.dump().c_str());
  if (config.contains("snapshot_interval")) {
    config_.snapshot_interval = config.at("snapshot_interval");
  }
  if (config.contains("min_snapshot_size")) {
    config_.min_snapshot_size = config.at("min_snapshot_size");
  }
  if (config.contains("crop_size_min")) {
    config_.crop_size_min = config.at("crop_size_min");
  }
  if (config.contains("crop_size_max")) {
    config_.crop_size_max = config.at("crop_size_max");
  }
  if (config.contains("snapshot_quality_threshold")) {
    config_.snapshot_quality_threshold =
        config.at("snapshot_quality_threshold");
  }
  if (config.contains("crop_square")) {
    config_.crop_square = config.at("crop_square");
  }
  if (config.contains("max_miss_counter")) {
    config_.max_miss_counter = config.at("max_miss_counter");
  }

  return 0;
}
int32_t ObjectSnapshot::updateSnapshot(
    std::shared_ptr<BaseImage> image, uint64_t frame_id,
    const std::map<uint64_t, ObjectBoxInfo>& track_boxes,
    const std::vector<TrackerInfo>& tracks,
    const std::map<uint64_t, float>& quality_scores,
    const std::map<std::string, Packet>& other_info,
    const std::map<uint64_t, std::shared_ptr<BaseImage>>& crop_face_imgs) {
  std::map<uint64_t, int> track_valid_flag;
  LOGI("ObjectSnapshot updateSnapshot, frame_id: %lu, tracks.size(): %zu",
       frame_id, tracks.size());
  for (size_t i = 0; i < tracks.size(); i++) {
    const TrackerInfo& track = tracks[i];
    track_valid_flag[track.track_id_] = 1;
    int obj_idx = track.obj_idx_;
    LOGI("process track_id: %lu, obj_idx: %d,i:%zu", track.track_id_, obj_idx,
         i);
    if (snapshot_infos_.count(track.track_id_)) {
      snapshot_infos_[track.track_id_].miss_counter++;
    }
    if (obj_idx == -1) {
      continue;
    }
    if (track_boxes.count(track.track_id_) == 0 ||
        quality_scores.count(track.track_id_) == 0) {
      LOGE(
          "track_boxes or quality_scores not found, track_id: "
          "%lu",
          track.track_id_);
      assert(false);
    }
    ObjectBoxInfo box = track_boxes.at(track.track_id_);
    float quality_score = quality_scores.at(track.track_id_);
    bool update_snapshot = false;
    if (snapshot_infos_.count(track.track_id_)) {
      if (quality_score > snapshot_infos_[track.track_id_].quality) {
        update_snapshot = true;
      }
    } else if (quality_score > config_.snapshot_quality_threshold) {
      update_snapshot = true;
      ObjectSnapshotInfo snapshot_info;
      // memset(&snapshot_info, 0, sizeof(ObjectSnapshotInfo));
      snapshot_infos_[track.track_id_] = snapshot_info;
    }
    if (update_snapshot) {
      ObjectSnapshotInfo& snapshot_info = snapshot_infos_[track.track_id_];
      snapshot_info.miss_counter = 0;

      snapshot_info.track_id = track.track_id_;
      snapshot_info.snapshot_frame_id = frame_id;
      snapshot_info.quality = quality_score;

      if (crop_face_imgs.count(track.track_id_)) {
        snapshot_info.object_image = crop_face_imgs.at(track.track_id_);
        if (snapshot_info.object_image == nullptr) {
          LOGE("ObjectSnapshot updateSnapshot, object_image is nullptr");
          assert(false);
        }
        LOGI(
            "update snapshot,track_id: %lu, quality: %f,image_width: "
            "%d,image_height: %d",
            track.track_id_, quality_score,
            snapshot_info.object_image->getWidth(),
            snapshot_info.object_image->getHeight());

        snapshot_info.object_box_info = box;

        if (other_info.count("face_landmark")) {
          const std::vector<ObjectBoxLandmarkInfo>& face_landmarks =
              other_info.at("face_landmark")
                  .get<std::vector<ObjectBoxLandmarkInfo>>();
          ObjectBoxLandmarkInfo landmark_info = face_landmarks[obj_idx];
          if (landmark_info.landmarks_x.size() != 5) {
            printf("landmark_info.landmarks_x size err : %ld\n",
                   landmark_info.landmarks_x.size());
            assert(false);
          }
          std::vector<float> landmarks;

          for (size_t i = 0; i < landmark_info.landmarks_x.size(); i++) {
            landmarks.push_back(landmark_info.landmarks_x[i]);
            landmarks.push_back(landmark_info.landmarks_y[i]);
          }
          snapshot_info.other_info["landmarks"] = Packet::make(landmarks);
        }
      } else {
        int crop_x = 0;
        int crop_y = 0;
        int crop_width = 0;
        int crop_height = 0;
        int dst_width = 0;
        int dst_height = 0;
        getCropBox(box, crop_x, crop_y, crop_width, crop_height, dst_width,
                   dst_height, image->getWidth(), image->getHeight());
        LOGI(
            "crop_x: %d, crop_y: %d, crop_width: %d, crop_height: %d, "
            "dst_width: "
            "%d, dst_height: %d,track_id: %lu",
            crop_x, crop_y, crop_width, crop_height, dst_width, dst_height,
            track.track_id_);
        snapshot_info.object_image =
            preprocessor_->cropResize(image, crop_x, crop_y, crop_width,
                                      crop_height, dst_width, dst_height);
        if (snapshot_info.object_image == nullptr) {
          LOGE("ObjectSnapshot updateSnapshot, object_image is nullptr");
          assert(false);
        }
        LOGI(
            "update snapshot,track_id: %lu, quality: %f,image_width: "
            "%d,image_height: %d",
            track.track_id_, quality_score,
            snapshot_info.object_image->getWidth(),
            snapshot_info.object_image->getHeight());
        snapshot_info.object_box_info = box;  // box is updated

        // TODO(fuquan.ke):wrap the code below as a callback function
        if (other_info.count("face_landmark")) {
          const std::vector<ObjectBoxLandmarkInfo>& face_landmarks =
              other_info.at("face_landmark")
                  .get<std::vector<ObjectBoxLandmarkInfo>>();
          ObjectBoxLandmarkInfo landmark_info = face_landmarks[obj_idx];
          std::vector<float> landmarks;
          float scale_x = dst_width / float(crop_width);
          float scale_y = dst_height / float(crop_height);
          if (landmark_info.landmarks_x.size() != 5) {
            printf("landmark_info.landmarks_x size err : %ld\n",
                   landmark_info.landmarks_x.size());
            assert(false);
          }
          for (size_t i = 0; i < landmark_info.landmarks_x.size(); i++) {
            landmarks.push_back((landmark_info.landmarks_x[i] - crop_x) *
                                scale_x);
            landmarks.push_back((landmark_info.landmarks_y[i] - crop_y) *
                                scale_y);
          }
          snapshot_info.other_info["landmarks"] = Packet::make(landmarks);
        }
      }
    }
  }
  LOGI(
      "ObjectSnapshot to update export_snapshots_, snapshot_infos_.size(): %zu",
      snapshot_infos_.size());
  for (auto iter = snapshot_infos_.begin(); iter != snapshot_infos_.end();) {
    if (track_valid_flag.count(iter->first) == 0 ||
        iter->second.miss_counter > config_.max_miss_counter) {
      // need to erase
      if (iter->second.object_image != nullptr) {
        export_snapshots_.push_back(iter->second);
      }
      iter = snapshot_infos_.erase(iter);
    } else {
      // check snapshot interval
      if (frame_id - iter->second.export_frame_id > static_cast<uint64_t>(config_.snapshot_interval)) {
        if (iter->second.object_image != nullptr) {
          LOGI(" to export ObjectSnapshotInfo\n");
          export_snapshots_.push_back(iter->second);
          resetSnapshotInfo(iter->second, frame_id);
        }
      }
      ++iter;
    }
  }
  LOGI(" export_snapshots_.size(): %zu", export_snapshots_.size());
  return 0;
}

int32_t ObjectSnapshot::getSnapshotData(
    std::vector<ObjectSnapshotInfo>& snapshots, bool force_all) {
  if (force_all) {
    for (auto iter = snapshot_infos_.begin(); iter != snapshot_infos_.end();) {
      if (iter->second.object_image != nullptr) {
        export_snapshots_.push_back(iter->second);
        iter = snapshot_infos_.erase(iter);
      } else {
        iter++;
      }
    }
  }
  snapshots = export_snapshots_;
  export_snapshots_.clear();
  LOGI("ObjectSnapshot getSnapshotData, snapshots.size(): %zu",
       snapshots.size());
  return 0;
}

void ObjectSnapshot::resetSnapshotInfo(ObjectSnapshotInfo& info,
                                       uint64_t frame_id) {
  info.miss_counter = 0;
  info.export_frame_id = frame_id;
  info.object_image = nullptr;
  info.quality = 0;
}
int32_t ObjectSnapshot::getCropBox(ObjectBoxInfo& box, int& x, int& y,
                                   int& width, int& height, int& dst_width,
                                   int& dst_height, int img_width,
                                   int img_height) {
  // 1. 计算原始框的宽高和中心
  float orig_w = box.x2 - box.x1;
  float orig_h = box.y2 - box.y1;
  float cx = box.x1 + orig_w * 0.5f;
  float cy = box.y1 + orig_h * 0.5f;

  // 2. 决定裁剪区域的尺寸（原图坐标下）
  //    如果要方形，就取 max(orig_w,orig_h)；否则保持原始长宽
  int crop_w = config_.crop_square ? int(std::max(orig_w, orig_h) + 0.5f)
                                   : int(orig_w + 0.5f);
  int crop_h = config_.crop_square ? crop_w : int(orig_h + 0.5f);

  // 3. 如果有最小裁剪限度，也可以在这里保证不小于它
  float scale = 1.0f;
  if (config_.crop_size_min > 0) {
    int max_crop_size = std::max(crop_w, crop_h);

    if (max_crop_size < config_.crop_size_min) {
      scale = float(config_.crop_size_min) / max_crop_size;
    } else {
      scale = float(config_.crop_size_max) / max_crop_size;
    }
    dst_width = int(crop_w * scale + 0.5f);
    dst_height = int(crop_h * scale + 0.5f);
  }

  // 4. 以中心为基准，计算左上角
  x = int(cx - crop_w * 0.5f + 0.5f);
  y = int(cy - crop_h * 0.5f + 0.5f);

  // 5. 边界控制：保证整个裁剪框在图像内部
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x + crop_w > img_width) x = std::max(0, img_width - crop_w);
  if (y + crop_h > img_height) y = std::max(0, img_height - crop_h);

  // 6. 最终输出
  width = crop_w;
  height = crop_h;
  box.x1 = (box.x1 - x) * scale;
  box.y1 = (box.y1 - y) * scale;
  box.x2 = (box.x2 - x) * scale;
  box.y2 = (box.y2 - y) * scale;
  return 0;
}
