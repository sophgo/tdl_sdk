#include "vehicle_adas.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include "framework/utils/tdl_log.hpp"

#define AVG_CAR_HEIGHT 1.4
#define AVG_BUS_HEIGHT 2.6
#define AVG_TRUCK_HEIGHT 3.5
#define AVG_RIDER_HEIGHT 1.4
#define AVG_PERSON_HEIGHT 1.6
#define AVG_DEFAULT_HEIGHT 1.2

#define ALPHA_CAR 0.475
#define ALPHA_MOTORBIKE 1.1

static float E = 2.7182818183;

int VehicleAdas::updateEasyQueue(int *data, int new_val, int size) {
  int sum = 0;
  for (int i = 0; i < size - 1; i++) {
    data[i] = data[i + 1];
    sum += data[i];
  }
  data[size - 1] = new_val;
  sum += new_val;
  return sum;
}

VehicleAdas::VehicleAdas() {
  config_.sence_type = 0;
  config_.location_type = 1;
  config_.miss_time_limit = 10;
  config_.collison_time = 4.0f;
  config_.departure_time = 1.0f;
  config_.FPS = 15.0f;
  config_.det_type = 0;
  config_.lane_model_type = 0;
}

VehicleAdas::~VehicleAdas() { deinit(); }

int32_t VehicleAdas::init(uint32_t buffer_size, int det_type) {
  LOGI("[APP::VehicleAdas] Initialize (Buffer Size: %u)\n", buffer_size);
  size_ = buffer_size;
  config_.det_type = det_type;
  data_.resize(buffer_size);
  return 0;
}

int32_t VehicleAdas::deinit() {
  LOGI("[APP::VehicleAdas] Deinit\n");
  data_.clear();
  size_ = 0;
  return 0;
}

int32_t VehicleAdas::updateConfig(const nlohmann::json &config) {
  LOGI("VehicleAdas update config");
  if (config.contains("sence_type")) {
    config_.sence_type = config.at("sence_type");
  }
  if (config.contains("location_type")) {
    config_.location_type = config.at("location_type");
  }
  if (config.contains("collison_time")) {
    config_.collison_time = config.at("collison_time");
  }
  if (config.contains("departure_time")) {
    config_.departure_time = config.at("departure_time");
  }
  if (config.contains("FPS")) {
    config_.FPS = config.at("FPS");
  }
  if (config.contains("miss_time_limit")) {
    config_.miss_time_limit = config.at("miss_time_limit");
  }
  return 0;
}

int32_t VehicleAdas::cleanData() {
  for (uint32_t j = 0; j < size_; j++) {
    if (data_[j].t_state == AdasTrackerState::MISS) {
      LOGI("[APP::VehicleAdas] Clean Vehicle Info[%u]\n", j);
      data_[j].info = ObjectBoxInfo();
      data_[j].t_state = AdasTrackerState::IDLE;
      data_[j].counter = 0;
      data_[j].miss_counter = 0;
      data_[j].dis = 0;
      data_[j].dis_tmp = 0;
      data_[j].start_score = 0;
      data_[j].warning_score = 0;
    }
  }
  return 0;
}

float VehicleAdas::objDis(const ObjectBoxInfo &info, float height,
                          float alpha) const {
  float obj_height;
  switch (info.class_id) {
    case 0:
      obj_height = AVG_CAR_HEIGHT;
      break;
    case 1:
      obj_height = AVG_BUS_HEIGHT;
      break;
    case 2:
      obj_height = AVG_TRUCK_HEIGHT;
      break;
    case 3:
      obj_height = AVG_RIDER_HEIGHT;
      break;
    case 4:
      obj_height = AVG_PERSON_HEIGHT;
      break;
    default:
      obj_height = AVG_DEFAULT_HEIGHT;
  }

  float dis = alpha * obj_height * height / (info.y2 - info.y1);

  if (info.y2 / height > 0.83) {
    dis *= 0.6;
  }

  return dis;
}

float VehicleAdas::objDisSelfCar(const ObjectBoxInfo &info, float height,
                                 float width) const {
  float h_ratio = (info.y2 - info.y1) / height;

  if (info.class_id > 4) {  // bike, motorbike
    return objDis(info, height, ALPHA_CAR);
  } else if (info.class_id == 3) {  // rider
    return -28.4 * h_ratio + 11.71;
  } else if (info.class_id == 4) {  // person
    return -19.75 * h_ratio + 10.08;
  } else if (info.class_id == 0) {  // car
    return -31.86 * h_ratio + 11.4;
  } else {
    float w_ratio = (info.x2 - info.x1) / width;
    float w_center = (info.x2 + info.x1) / (2.0f * width);
    float w_h_ratio = (info.x2 - info.x1) / (info.y2 - info.y1);

    if (info.class_id == 1) {  // bus
      if (w_center > 0.4 && w_center < 0.6 && w_h_ratio < 1.3) {
        return 15.91 * pow(E, -5.019 * w_ratio);
      }
      return objDis(info, height, ALPHA_CAR);
    } else {  // truck
      if (w_center > 0.4 && w_center < 0.6 && w_h_ratio < 1.4) {
        return 14.553 * pow(E, -4.746 * w_ratio);
      }
      return objDis(info, height, ALPHA_CAR);
    }
  }
}

float VehicleAdas::objDisSelfMotorbike(const ObjectBoxInfo &info, float height,
                                       float width, bool center) const {
  float h_ratio = (info.y2 - info.y1) / height;

  if (info.class_id > 4) {  // bike, motorbike
    return objDis(info, height, ALPHA_MOTORBIKE);
  } else if (info.class_id == 4) {  // person
    return -7.87 * h_ratio + 8.11;
  } else {
    float w_ratio = (info.x2 - info.x1) / width;
    float w_h_ratio = (info.x2 - info.x1) / (info.y2 - info.y1);

    if (info.class_id == 3) {  // rider
      if (center && w_h_ratio < 0.5) {
        return 0.0961 * pow(w_ratio, -1.568);
      }
      return objDis(info, height, ALPHA_MOTORBIKE);
    } else if (info.class_id == 0) {  // car
      if (center && w_h_ratio < 1.3) {
        return 0.604 * pow(w_ratio, -1.483);
      }
      return objDis(info, height, ALPHA_MOTORBIKE);
    } else {  // truck, bus
      if (center && w_h_ratio < 1.3) {
        return 10.668 * pow(E, -2.43 * w_ratio);
      }
      return objDis(info, height, ALPHA_MOTORBIKE);
    }
  }
}

void VehicleAdas::updateDis(ObjectBoxInfo &info, VehicleAdasObjectData &data,
                            int obj_index, float height, float width,
                            bool first_time) {
  float cur_dis;
  bool center = center_info_[0] == obj_index;

  if (config_.sence_type == 0) {
    cur_dis = objDisSelfCar(info, height, width);
  } else {
    cur_dis = objDisSelfMotorbike(info, height, width, center);
  }

  if (cur_dis < 0) {
    cur_dis = 0.5;
  }

  if (!first_time) {
    cur_dis = 0.9 * data.dis + 0.1 * cur_dis;
  } else {
    data.dis_tmp = cur_dis;
  }

  data.dis = cur_dis;
}

void VehicleAdas::updateSpeed(ObjectBoxInfo &info, VehicleAdasObjectData &data,
                              float fps) {
  if (data.counter >= 3) {
    float cur_speed =
        (data.dis - data.dis_tmp) / ((float)(data.counter - 1) / fps);
    data.speed = cur_speed;
    data.dis_tmp = data.dis;
    data.counter = 0;
  }
}

void VehicleAdas::updateObjectState(ObjectBoxInfo &info,
                                    VehicleAdasObjectData &data, uint32_t width,
                                    int obj_index, bool self_static) {
  float cur_start_score = 0;
  float cur_warning_score = 0;

  if (self_static && center_info_[0] == obj_index && data.dis < 4.0f) {
    if (data.speed > 0.3) {
      cur_start_score = 1.0f;
    } else if (data.speed > 0.1) {
      cur_start_score = (data.speed - 0.1) / 0.3;
    }
  }

  data.start_score = 0.9 * data.start_score + 0.1 * cur_start_score;

  bool center = center_info_[0] == obj_index;

  if (center && data.dis < 8.0 &&
      -(data.speed) * (float)config_.collison_time > data.dis) {
    cur_warning_score = 1.0f;
  }
  data.warning_score = 0.9 * data.warning_score + 0.1 * cur_warning_score;
}

void VehicleAdas::updateFrameState() {
  if (gsensor_data_.counter > gsensor_tmp_data_.counter) {
    if (gsensor_data_.counter > 1) {
      int diff = abs(gsensor_data_.x - gsensor_tmp_data_.x) +
                 abs(gsensor_data_.y - gsensor_tmp_data_.y) +
                 abs(gsensor_data_.z - gsensor_tmp_data_.z);

      int gsensor_period_sum = updateEasyQueue(gsensor_queue_, diff, 10);

      if (gsensor_period_sum > 22) {
        is_static_ = false;
      } else {
        is_static_ = true;
      }
    }
    gsensor_tmp_data_ = gsensor_data_;
  }
}

void VehicleAdas::frontObjIndex(
    const std::vector<ObjectBoxInfo> &object_infos,
    const std::shared_ptr<ModelBoxLandmarkInfo> &lane_meta, float width) {
  center_info_[0] = -1;
  float dis = width;

  float left_x = 0.4f * width;
  float right_x = 0.6f * width;

  if (config_.sence_type) {  // motorbike
    left_x = left_x - ((float)config_.location_type - 1.0f) * 0.1f * width;
    right_x = right_x - ((float)config_.location_type - 1.0f) * 0.1f * width;
  }

  if (config_.det_type && lane_meta && lane_meta->box_landmarks.size() == 2) {
    float xmin = lane_meta->box_landmarks[0].landmarks_y[0] <
                         lane_meta->box_landmarks[0].landmarks_y[1]
                     ? lane_meta->box_landmarks[0].landmarks_x[0]
                     : lane_meta->box_landmarks[0].landmarks_x[1];
    float xmax = lane_meta->box_landmarks[1].landmarks_y[0] <
                         lane_meta->box_landmarks[1].landmarks_y[1]
                     ? lane_meta->box_landmarks[1].landmarks_x[0]
                     : lane_meta->box_landmarks[1].landmarks_x[1];

    if (xmin > xmax) {
      float tmp = xmin;
      xmin = xmax;
      xmax = tmp;
    }

    if (xmax - xmin > 0.15f * width && xmax - xmin < 0.35f * width) {
      left_x = xmin;
      right_x = xmax;
    }
  }

  center_info_[1] = left_x;
  center_info_[2] = right_x;

  float center = (left_x + right_x) / 2.0f;

  for (size_t i = 0; i < object_infos.size(); i++) {
    if (object_infos[i].class_id < 5) {
      float c_x = (object_infos[i].x1 + object_infos[i].x2) / 2.0f;

      if (c_x > left_x && c_x < right_x) {
        float cur_dis = center > c_x ? center - c_x : c_x - center;

        if (cur_dis < dis) {
          center_info_[0] = i;
          dis = cur_dis;
        }
      }
    }
  }
}

int32_t VehicleAdas::updateData(uint32_t frame_width, uint32_t frame_height,
                                const std::vector<ObjectBoxInfo> &object_infos,
                                const std::vector<TrackerInfo> &track_results) {
  LOGI("[APP::VehicleAdas] Update Data, objects:%zu, tracks:%zu\n",
       object_infos.size(), track_results.size());

  for (size_t i = 0; i < track_results.size(); i++) {
    const TrackerInfo &track = track_results[i];
    uint64_t trk_id = track.track_id_;
    int obj_idx = track.obj_idx_;

    if (obj_idx < 0 || obj_idx >= (int)object_infos.size()) {
      continue;
    }

    // Find existing entry by track_id (proper matching)
    int match_idx = -1;
    int idle_idx = -1;

    for (uint32_t j = 0; j < size_; j++) {
      if (data_[j].t_state == AdasTrackerState::ALIVE &&
          data_[j].track_id == trk_id) {
        match_idx = (int)j;
        break;
      }
    }

    if (match_idx == -1) {
      for (uint32_t j = 0; j < size_; j++) {
        if (data_[j].t_state == AdasTrackerState::IDLE) {
          idle_idx = (int)j;
          break;
        }
      }
    }

    int update_idx = (match_idx != -1) ? match_idx : idle_idx;

    if (update_idx == -1) {
      LOGD("no valid buffer for track %lu\n", trk_id);
      continue;
    }

    data_[update_idx].track_id = trk_id;
    data_[update_idx].info = object_infos[obj_idx];
    data_[update_idx].t_state = AdasTrackerState::ALIVE;
    data_[update_idx].miss_counter = 0;
    data_[update_idx].counter += 1;

    updateDis(data_[update_idx].info, data_[update_idx], obj_idx, frame_height,
              frame_width, match_idx == -1);

    if (match_idx != -1) {
      updateSpeed(data_[update_idx].info, data_[update_idx], config_.FPS);
      updateObjectState(data_[update_idx].info, data_[update_idx], frame_width,
                        obj_idx, is_static_);
    }
  }

  // Mark missing tracks (those not seen in current frame)
  for (uint32_t j = 0; j < size_; j++) {
    if (data_[j].t_state != AdasTrackerState::ALIVE) continue;

    bool found = false;
    for (size_t k = 0; k < track_results.size(); k++) {
      if (data_[j].track_id == track_results[k].track_id_ &&
          track_results[k].obj_idx_ >= 0) {
        found = true;
        break;
      }
    }

    if (!found) {
      data_[j].miss_counter += 1;
      data_[j].counter += 1;

      if (data_[j].miss_counter >= config_.miss_time_limit) {
        data_[j].t_state = AdasTrackerState::MISS;
      }
    }
  }

  return 0;
}

int32_t VehicleAdas::updateLaneState(
    const std::shared_ptr<ModelBoxLandmarkInfo> &lane_meta, uint32_t height,
    uint32_t width) {
  if (lane_meta == nullptr) return 0;

  float cur_score = 0;

  for (size_t i = 0; i < lane_meta->box_landmarks.size(); i++) {
    float x0 = lane_meta->box_landmarks[i].landmarks_x[0];
    float y0 = lane_meta->box_landmarks[i].landmarks_y[0];
    float x1 = lane_meta->box_landmarks[i].landmarks_x[1];
    float y1 = lane_meta->box_landmarks[i].landmarks_y[1];

    float k = (y1 - y0) / (x1 - x0);
    k = k > 0 ? k : -k;
    float x_i = ((float)height * 0.78 - y0) / (y1 - y0) * (x1 - x0) + x0;

    if (x_i > (float)width * 0.3 && x_i < (float)width * 0.7 && k > 1.2) {
      cur_score = 1.0f;
      break;
    }
  }

  lane_state_.lane_score = 0.95f * lane_state_.lane_score + 0.05f * cur_score;
  float departure_time = config_.departure_time - 0.4f;
  if (lane_state_.lane_score > config_.FPS / 50.0f) {
    lane_state_.lane_counter++;
    if ((float)lane_state_.lane_counter / config_.FPS > departure_time) {
      lane_state_.lane_state = 1;
    } else {
      lane_state_.lane_state = 0;
    }
  } else {
    lane_state_.lane_counter = 0;
    lane_state_.lane_state = 0;
  }

  return 0;
}

void VehicleAdas::getResults(std::vector<VehicleAdasObjectResult> &results) {
  results.clear();
  for (uint32_t j = 0; j < size_; j++) {
    if (data_[j].t_state == AdasTrackerState::ALIVE &&
        data_[j].miss_counter == 0) {
      VehicleAdasObjectResult result;
      result.track_id = data_[j].track_id;
      result.info = data_[j].info;
      result.distance = data_[j].dis;
      result.speed = data_[j].speed;

      if (data_[j].start_score > config_.FPS / 30.0f) {
        result.state = AdasState::START;
      } else if (data_[j].warning_score > config_.FPS / 30.0f) {
        result.state = AdasState::COLLISION_WARNING;
      } else {
        result.state = AdasState::NORMAL;
      }

      results.push_back(result);
    }
  }
}

// When relevant data is available, a new C interface can be added to update it
void VehicleAdas::updateGsensor(float x, float y, float z) {
  gsensor_data_.x = (int)x;
  gsensor_data_.y = (int)y;
  gsensor_data_.z = (int)z;
  gsensor_data_.counter += 1;
}

int32_t VehicleAdas::run(
    uint64_t frame_id, uint32_t frame_width, uint32_t frame_height,
    const std::vector<ObjectBoxInfo> &object_infos,
    const std::vector<TrackerInfo> &track_results,
    const std::shared_ptr<ModelBoxLandmarkInfo> &lane_meta) {
  if (size_ == 0) {
    LOGE("[APP::VehicleAdas] not initialized\n");
    return -1;
  }

  cleanData();

  if (config_.det_type && lane_meta) {
    updateLaneState(lane_meta, frame_height, frame_width);
    updateFrameState();
  }

  frontObjIndex(object_infos, lane_meta, frame_width);

  updateData(frame_width, frame_height, object_infos, track_results);

  return 0;
}