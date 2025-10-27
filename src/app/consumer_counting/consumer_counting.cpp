#include "consumer_counting.hpp"
#include <cstdio>
#include <json.hpp>
#include "app/app_data_types.hpp"

#include "utils/tdl_log.hpp"

ConsumerCounting::ConsumerCounting(int A_x, int A_y, int B_x, int B_y, int mode,
                                   int counting_gap) {
  counting_gap_ = counting_gap;
  set_counting_line(A_x, A_y, B_x, B_y, mode);
}

uint32_t ConsumerCounting::set_counting_line(int A_x, int A_y, int B_x, int B_y,
                                             int mode) {
  A_x_ = A_x;
  A_y_ = A_y;
  B_x_ = B_x;
  B_y_ = B_y;
  mode_ = mode;

  // 对于竖直线，从左到右为进入，对于非竖直线，从上到下为进入

  assert(mode == 0 || mode == 1 || mode == 2);
  if (mode == 2) {
    mode = 0;
  }

  int dx = B_x_ - A_x_;
  int dy = B_y_ - A_y_;

  if (dx == 0) {
    if (mode == 0) {
      normal_vector_x_ = 1.0;
      normal_vector_y_ = 0;
    } else {
      normal_vector_x_ = -1.0;
      normal_vector_y_ = 0;
    }
  } else {
    if ((dx > 0 && mode == 0) || (dx < 0 && mode == 1)) {
      normal_vector_x_ = (float)-dy;
      normal_vector_y_ = (float)dx;
    } else {
      normal_vector_x_ = (float)dy;
      normal_vector_y_ = (float)-dx;
    }
  }

  LOGI("normal_vector: x: %f. y: %f\n", normal_vector_x_, normal_vector_y_);

  return 0;
}

uint32_t ConsumerCounting::get_counting_line(std::vector<int> &counting_line) {
  counting_line.push_back(A_x_);
  counting_line.push_back(A_y_);
  counting_line.push_back(B_x_);
  counting_line.push_back(B_y_);
  return 0;
}

float ConsumerCounting::crossProduct(float A_x, float A_y, float B_x, float B_y,
                                     float C_x, float C_y) {
  return (B_x - A_x) * (C_y - A_y) - (B_y - A_y) * (C_x - A_x);
}

bool ConsumerCounting::isLineIntersect(float old_x, float old_y, float cur_x,
                                       float cur_y) {
  double cp1 = crossProduct(A_x_, A_y_, B_x_, B_y_, old_x, old_y);
  double cp2 = crossProduct(A_x_, A_y_, B_x_, B_y_, cur_x, cur_y);

  double cp3 = crossProduct(old_x, old_y, cur_x, cur_y, A_x_, A_y_);
  double cp4 = crossProduct(old_x, old_y, cur_x, cur_y, B_x_, B_y_);
  // positive and negative symbols
  if ((cp1 * cp2 <= 0) && (cp3 * cp4 <= 0)) return true;
  return false;
}

int32_t ConsumerCounting::consumer_counting(float old_x, float old_y,
                                            float cur_x, float cur_y,
                                            Consumer &it) {
  if (isLineIntersect(old_x, old_y, cur_x, cur_y)) {
    float tmp_x = cur_x - old_x;
    float tmp_y = cur_y - old_y;
    if ((tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ < 0) &&
        it.counting_gap == counting_gap_ && !it.miss) {
      miss_num_++;
      it.miss = true;
    } else if ((tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ > 0) &&
               it.counting_gap == counting_gap_ && !it.enter) {
      enter_num_++;
      it.enter = true;
    }
  }

  it.counting_gap = 0;

  return 0;
}

bool ConsumerCounting::object_cross(float old_x, float old_y, float cur_x,
                                    float cur_y, Consumer &it) {
  bool is_cross = false;
  if (isLineIntersect(old_x, old_y, cur_x, cur_y)) {
    float tmp_x = cur_x - old_x;
    float tmp_y = cur_y - old_y;

    if ((tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ > 0) &&
        !it.is_cross) {
      is_cross = true;
    } else if (mode_ == 2 &&
               (tmp_x * normal_vector_x_ + tmp_y * normal_vector_y_ < 0) &&
               !it.is_cross) {
      is_cross = true;
    }
  }
  it.counting_gap = 0;

  return is_cross;
}

int32_t ConsumerCounting::update_consumer_counting_state(
    const std::vector<TrackerInfo> &track_results, bool force_all) {
  if (force_all) {  // for evaluation
    printf("force_all!");
    for (auto it = heads_.begin(); it != heads_.end();) {
      consumer_counting(it->second.old_x, it->second.old_y, it->second.new_x,
                        it->second.new_y, it->second);
      LOGI(
          "to count heads, track id: %ld, old_x: %.2f, old_y: %.2f, cur_x: "
          "%.2f, cur_y: %.2f, enter: %d, miss: %d\n",
          it->first, it->second.old_x, it->second.old_y, it->second.new_x,
          it->second.new_y, it->second.enter, it->second.miss);
      it++;
    }
    return 0;
  }

  std::map<uint64_t, int> head_index;
  std::map<uint64_t, int> person_index;
  std::vector<uint64_t> head_found_id;
  std::vector<uint64_t> person_found_id;

  for (size_t i = 0; i < track_results.size(); i++) {
    const TrackerInfo &t = track_results[i];

    uint64_t track_id = t.track_id_;

    if (t.box_info_.object_type == TDLObjectType::OBJECT_TYPE_HEAD) {
      head_index[track_id] = i;
    } else {
      if (t.box_info_.object_type != TDLObjectType::OBJECT_TYPE_PERSON) {
        LOGE("unexpected object_type: %d\n", t.box_info_.object_type);
        assert(false);
        return -1;
      }
      person_index[track_id] = i;
    }
  }

  LOGI("head index size: %d, person index size: %d\n", head_index.size(),
       person_index.size());

  LOGI("heads_ size: %d\n", heads_.size());
  for (auto it = heads_.begin(); it != heads_.end();) {
    if (head_index.count(it->first) == 0) {
      LOGI("track id: %ld, it->second.unmatched_times: %d\n", it->first,
           it->second.unmatched_times);
      it->second.unmatched_times += 1;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      if (it->second.unmatched_times == MAX_UNMATCHED_TIME) {
        consumer_counting(it->second.old_x, it->second.old_y, it->second.new_x,
                          it->second.new_y, it->second);
        LOGI(
            "to count heads before erasing, track id: %ld, old_x: %.2f, old_y: "
            "%.2f, cur_x: "
            "%.2f, cur_y: %.2f, enter: %d, miss: %d\n",
            it->first, it->second.old_x, it->second.old_y, it->second.new_x,
            it->second.new_y, it->second.enter, it->second.miss);

        LOGI("erase head track id: %ld\n", it->first);

        it = heads_.erase(it);
      } else {
        it++;
      }

    } else {
      const TrackerInfo &track_info = track_results[head_index[it->first]];
      uint64_t pair_id = track_info.pair_track_idx_;

      ObjectBoxInfo box_info = track_info.box_info_;
      float cur_x = (box_info.x1 + box_info.x2) / 2.0;
      float cur_y = (box_info.y1 + box_info.y2) / 2.0;
      it->second.new_x = cur_x;
      it->second.new_y = cur_y;

      if (it->second.enter && it->second.miss) {
        LOGI("track id %ld has been counted\n", it->first);
      } else if (it->second.counting_gap == counting_gap_) {
        consumer_counting(it->second.old_x, it->second.old_y, cur_x, cur_y,
                          it->second);
        LOGI(
            "to count heads, track id: %ld, old_x: %.2f, old_y: %.2f, cur_x: "
            "%.2f, cur_y: %.2f, enter: %d, miss: %d\n",
            it->first, it->second.old_x, it->second.old_y, cur_x, cur_y,
            it->second.enter, it->second.miss);
        it->second.old_x = cur_x;
        it->second.old_y = cur_y;
      }

      it->second.unmatched_times = 0;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      head_found_id.push_back(it->first);
      it++;
    }
  }

  for (auto it = head_index.begin(); it != head_index.end();) {
    if (std::find(head_found_id.begin(), head_found_id.end(), it->first) ==
        head_found_id.end()) {
      const TrackerInfo &t = track_results[it->second];
      ObjectBoxInfo box_info = t.box_info_;

      uint64_t track_id = t.track_id_;

      Consumer new_consumer = {0};
      new_consumer.old_x = (box_info.x1 + box_info.x2) / 2.0;
      new_consumer.old_y = (box_info.y1 + box_info.y2) / 2.0;

      LOGI("new head track id: %ld, old_x: %.2f, old_y: %.2f\n", track_id,
           new_consumer.old_x, new_consumer.old_y);
      heads_[track_id] = new_consumer;
    }
    it++;
  }

  return 0;
}

int32_t ConsumerCounting::update_cross_detection_state(
    const std::vector<TrackerInfo> &track_results,
    std::vector<uint64_t> &cross_id) {
  cross_id.clear();
  std::map<uint64_t, int> object_index;
  std::vector<uint64_t> found_id;

  for (size_t i = 0; i < track_results.size(); i++) {
    const TrackerInfo &t = track_results[i];

    uint64_t track_id = t.track_id_;
    object_index[track_id] = i;
  }

  for (auto it = objects_.begin(); it != objects_.end();) {
    if (object_index.count(it->first) == 0) {
      LOGI("track id: %ld, it->second.unmatched_times: %d\n", it->first,
           it->second.unmatched_times);
      it->second.unmatched_times += 1;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      if (it->second.unmatched_times == MAX_UNMATCHED_TIME) {
        LOGI("erase head track id: %ld\n", it->first);
        it = objects_.erase(it);
      }

    } else {
      const TrackerInfo &track_info = track_results[object_index[it->first]];
      uint64_t pair_id = track_info.pair_track_idx_;

      if (it->second.counting_gap == counting_gap_ &&
          track_info.obj_idx_ != -1) {
        ObjectBoxInfo box_info = track_info.box_info_;
        float cur_x = (box_info.x1 + box_info.x2) / 2.0;
        float cur_y = (box_info.y1 + box_info.y2) / 2.0;

        bool is_cross = object_cross(it->second.old_x, it->second.old_y, cur_x,
                                     cur_y, it->second);

        LOGI(
            "to detect, track id: %ld, old_x: %.2f, old_y: %.2f, cur_x: "
            "%.2f, cur_y: %.2f, is_cross: %d\n",
            it->first, it->second.old_x, it->second.old_y, cur_x, cur_y,
            is_cross);
        it->second.old_x = cur_x;
        it->second.old_y = cur_y;
        if (is_cross) {
          cross_id.push_back(it->first);
        }
      }

      it->second.unmatched_times = 0;
      it->second.counting_gap =
          std::min(counting_gap_, it->second.counting_gap + 1);

      found_id.push_back(it->first);
      it++;
    }
  }

  for (auto it = object_index.begin(); it != object_index.end();) {
    if (std::find(found_id.begin(), found_id.end(), it->first) ==
        found_id.end()) {
      const TrackerInfo &t = track_results[it->second];
      ObjectBoxInfo box_info = t.box_info_;

      uint64_t track_id = t.track_id_;

      Consumer new_consumer = {0};
      new_consumer.old_x = (box_info.x1 + box_info.x2) / 2.0;
      new_consumer.old_y = (box_info.y1 + box_info.y2) / 2.0;

      LOGI("new track id: %ld, old_x: %.2f, old_y: %.2f\n", track_id,
           new_consumer.old_x, new_consumer.old_y);
      objects_[track_id] = new_consumer;
    }
    it++;
  }

  return 0;
}