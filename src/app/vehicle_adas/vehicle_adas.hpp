#ifndef TDL_SDK_VEHICLE_ADAS_HPP
#define TDL_SDK_VEHICLE_ADAS_HPP

#include <cstdint>
#include <json.hpp>
#include <map>
#include <memory>
#include <vector>
#include "components/tracker/tracker_types.hpp"
#include "framework/common/model_output_types.hpp"
#include "framework/image/base_image.hpp"

enum class AdasState { NORMAL = 0, START = 1, COLLISION_WARNING = 2 };

enum class AdasTrackerState { IDLE = 0, ALIVE = 1, MISS = 2 };

struct VehicleAdasObjectData {
  uint64_t track_id = 0;
  ObjectBoxInfo info;
  AdasTrackerState t_state = AdasTrackerState::IDLE;
  float speed = 0.0f;
  float dis = 0.0f;
  float dis_tmp = 0.0f;
  float start_score = 0.0f;
  float warning_score = 0.0f;
  uint32_t miss_counter = 0;
  uint32_t counter = 0;
};

struct VehicleAdasObjectResult {
  uint64_t track_id = 0;
  ObjectBoxInfo info;
  float distance = 0.0f;
  float speed = 0.0f;
  AdasState state = AdasState::NORMAL;
};

struct VehicleAdasLaneLine {
  float x1, y1, x2, y2;
};

struct VehicleAdasConfig {
  int sence_type = 0;
  int location_type = 1;
  int det_type = 0;
  int lane_model_type = 0;
  uint32_t miss_time_limit = 10;
  float collison_time = 4.0f;
  float departure_time = 1.0f;
  float FPS = 15.0f;
};

struct VehicleAdasLaneState {
  int lane_state = 0;
  float lane_score = 0.0f;
  int lane_counter = 0;
};

struct VehicleAdasResult {
 public:
  uint64_t frame_id;
  uint32_t frame_width;
  uint32_t frame_height;
  std::shared_ptr<BaseImage> image;
  std::vector<VehicleAdasObjectResult> objects;
  VehicleAdasLaneState lane_state;
  std::vector<VehicleAdasLaneLine> lane_lines;
  std::vector<TrackerInfo> track_results;
};

class VehicleAdas {
 public:
  VehicleAdas();
  ~VehicleAdas();

  int32_t init(uint32_t buffer_size, int det_type);
  int32_t deinit();
  int32_t updateConfig(const nlohmann::json& config);
  void setConfig(const VehicleAdasConfig& config) { config_ = config; }
  const VehicleAdasConfig& getConfig() const { return config_; }
  const VehicleAdasLaneState& getLaneState() const { return lane_state_; }

  int32_t run(uint64_t frame_id, uint32_t frame_width, uint32_t frame_height,
              const std::vector<ObjectBoxInfo>& object_infos,
              const std::vector<TrackerInfo>& track_results,
              const std::shared_ptr<ModelBoxLandmarkInfo>& lane_meta);

  void getResults(std::vector<VehicleAdasObjectResult>& results);
  void updateGsensor(float x, float y, float z);

 private:
  void frontObjIndex(const std::vector<ObjectBoxInfo>& object_infos,
                     const std::shared_ptr<ModelBoxLandmarkInfo>& lane_meta,
                     float width);
  int32_t cleanData();
  int32_t updateData(uint32_t frame_width, uint32_t frame_height,
                     const std::vector<ObjectBoxInfo>& object_infos,
                     const std::vector<TrackerInfo>& track_results);

  // Distance estimation
  float objDis(const ObjectBoxInfo& info, float height, float alpha) const;
  float objDisSelfCar(const ObjectBoxInfo& info, float height,
                      float width) const;
  float objDisSelfMotorbike(const ObjectBoxInfo& info, float height,
                            float width, bool center) const;
  void updateDis(ObjectBoxInfo& info, VehicleAdasObjectData& data,
                 int obj_index, float height, float width, bool first_time);
  void updateSpeed(ObjectBoxInfo& info, VehicleAdasObjectData& data, float fps);
  void updateObjectState(ObjectBoxInfo& info, VehicleAdasObjectData& data,
                         uint32_t width, int obj_index, bool self_static);

  // Lane detection
  int32_t updateLaneState(
      const std::shared_ptr<ModelBoxLandmarkInfo>& lane_meta, uint32_t height,
      uint32_t width);

  // G-sensor
  void updateFrameState();
  int updateEasyQueue(int* data, int new_val, int size);

  VehicleAdasConfig config_;
  VehicleAdasLaneState lane_state_;
  std::vector<VehicleAdasObjectData> data_;
  uint32_t size_ = 0;

  int center_info_[3] = {-1, 0, 0};

  struct GsensorData {
    int x = 0, y = 0, z = 0;
    uint64_t counter = 0;
  };
  GsensorData gsensor_data_;
  GsensorData gsensor_tmp_data_;
  int gsensor_queue_[10] = {0};
  bool is_static_ = true;
};

#endif /* TDL_SDK_VEHICLE_ADAS_HPP */